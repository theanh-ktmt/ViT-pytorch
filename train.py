# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_datasets, get_loader, NUM_CLASS_MAPPING
from utils.k_fold import KFoldDataset, KFoldEnsembleModel


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, fold=0):
    model_to_save = model.module if hasattr(model, "module") else model
    model_checkpoint = os.path.join(
        args.output_dir, f"{args.name}_fold{fold}_checkpoint.bin"
    )
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    return model_checkpoint


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    num_classes = NUM_CLASS_MAPPING[args.dataset]

    model = VisionTransformer(
        config, args.img_size, zero_head=True, num_classes=num_classes
    )
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step, is_test=False):
    # Validation!
    eval_losses = AverageMeter()

    logger.info(
        "***** Running {} *****".format("Validation" if not is_test else "Testing")
    )
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(
        test_loader,
        desc="{}... (loss=X.X)".format("Validating" if not is_test else "Testing"),
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0],
    )
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    avg_loss = eval_losses.avg

    logger.info("\n")
    logger.info("{} Results".format("Test" if is_test else "Validation"))
    logger.info("Global Steps: %d" % global_step)
    if is_test:
        logger.info("Test Loss: %2.5f" % avg_loss)
        logger.info("Test Accuracy: %2.5f" % accuracy)
        writer.add_scalar(
            "test/accuracy", scalar_value=accuracy, global_step=global_step
        )
        writer.add_scalar("test/loss", scalar_value=avg_loss, global_step=global_step)
    else:
        logger.info("Valid Loss: %2.5f" % avg_loss)
        logger.info("Valid Accuracy: %2.5f" % accuracy)
        writer.add_scalar(
            "val/accuracy", scalar_value=accuracy, global_step=global_step
        )
        writer.add_scalar("val/loss", scalar_value=avg_loss, global_step=global_step)
    return accuracy, avg_loss


def train(args):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    trainset, testset = get_datasets(args)
    if args.k_fold > 1:
        trainset = KFoldDataset(args).split(trainset)
    test_loader = get_loader(testset, args, eval=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()

    fold_accs, fold_losses, fold_paths = [], [], []
    for fold in range(args.k_fold):
        logger.info(f"Training fold: {fold + 1} / {args.k_fold}")

        # Model & Tokenizer Setup
        args, model = setup(args)

        # Freeze all layers except the head
        print(model)
        for name, module in model.named_parameters():
            if name.startswith("head"):
                module.requires_grad = True
            else:
                module.requires_grad = False

        # Prepare optimizer and scheduler
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        t_total = args.num_steps
        if args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total
            )
        else:
            scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total
            )
        model.zero_grad()

        # prepare dataset
        if args.k_fold > 1:
            train_loader, val_loader = trainset.get_fold(fold)
        else:
            train_loader = get_loader(trainset, args)
            val_loader = test_loader

        global_step = 0
        best_acc, best_loss = 0, float("inf")
        best_path = None

        while True:
            model.train()
            epoch_iterator = tqdm(
                train_loader,
                desc="Training (X / X Steps) (loss=X.X)",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True,
                disable=args.local_rank not in [-1, 0],
            )
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                loss = model(x, y)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    pass
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses.update(loss.item() * args.gradient_accumulation_steps)
                    if args.fp16:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        pass
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)"
                        % (global_step, t_total, losses.val)
                    )
                    if args.local_rank in [-1, 0]:
                        writer.add_scalar(
                            "train/loss",
                            scalar_value=losses.val,
                            global_step=global_step,
                        )
                        writer.add_scalar(
                            "train/lr",
                            scalar_value=scheduler.get_lr()[0],
                            global_step=global_step,
                        )
                    if global_step % args.eval_every == 0 and args.local_rank in [
                        -1,
                        0,
                    ]:
                        val_acc, val_loss = valid(
                            args, model, writer, val_loader, global_step
                        )
                        if best_loss > val_loss:
                            best_path = save_model(args, model, fold)
                            logger.info("Best model saved at %s" % best_path)

                            logger.info(
                                "Update best loss from %f to %f" % (best_loss, val_loss)
                            )
                            logger.info(
                                "Update best accuracy from %f to %f"
                                % (best_acc, val_acc)
                            )
                            best_loss = val_loss
                            best_acc = val_acc
                        model.train()

                    if global_step % t_total == 0:
                        break

            losses.reset()
            if global_step % t_total == 0:
                break

        if args.local_rank in [-1, 0]:
            writer.close()
        logger.info("Best Accuracy: \t%f" % best_acc)
        logger.info("Best Loss: \t%f" % best_loss)
        logger.info("Done training fold %d" % (fold + 1))
        logger.info("===================================")

        # save fold results
        fold_accs.append(best_acc)
        fold_losses.append(best_loss)
        fold_paths.append(best_path)

    logger.info("All folds done!")
    logger.info("Average Accuracy: \t%f" % np.mean(fold_accs))
    logger.info("Average Loss: \t%f" % np.mean(fold_losses))
    logger.info("All Paths: %s" % fold_paths)
    logger.info("===================================")

    # test the best model
    if args.do_test:
        if args.k_fold > 1:
            model.load_state_dict(torch.load(best_path))
        else:
            model = KFoldEnsembleModel(fold_paths, args, decide_mode="mean")
        valid(args, model, writer, test_loader, global_step, is_test=True)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--name", required=True, help="Name of this run. Used for monitoring."
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "hymenoptera"],
        default="cifar10",
        help="Which downstream task.",
    )
    parser.add_argument(
        "--model_type",
        choices=[
            "ViT-B_16",
            "ViT-B_32",
            "ViT-L_16",
            "ViT-L_32",
            "ViT-H_14",
            "R50-ViT-B_16",
        ],
        default="ViT-B_16",
        help="Which variant to use.",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="checkpoint/ViT-B_16.npz",
        help="Where to search for pretrained ViT models.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where checkpoints will be written.",
    )

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--eval_every",
        default=100,
        type=int,
        help="Run prediction on validation set every so many steps."
        "Will always run one evaluation at the end of training.",
    )

    parser.add_argument(
        "--learning_rate",
        default=3e-2,
        type=float,
        help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--weight_decay", default=0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--num_steps",
        default=10000,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear"],
        default="cosine",
        help="How to decay the learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="Step of training to perform learning rate warmup for.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )

    # custom args
    parser.add_argument(
        "--k-fold", type=int, default=1, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--do-test",
        action="store_true",
        help="Whether to test the model after training",
    )

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(minutes=60)
        )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s"
        % (
            args.local_rank,
            args.device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )
    )

    # Set seed
    set_seed(args)

    # Training
    train(args)


if __name__ == "__main__":
    main()
