import logging

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


NUM_CLASS_MAPPING = {"cifar10": 10, "cifar100": 100, "hymenoptera": 2}


def get_datasets(args):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (args.img_size, args.img_size), scale=(0.05, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform_test
            )
            if args.local_rank in [-1, 0]
            else None
        )

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform_test
            )
            if args.local_rank in [-1, 0]
            else None
        )

    elif args.dataset == "hymenoptera":
        trainset = datasets.ImageFolder(
            root="./data/hymenoptera/train", transform=transform_train
        )
        testset = (
            datasets.ImageFolder(
                root="./data/hymenoptera/val", transform=transform_test
            )
            if args.local_rank in [-1, 0]
            else None
        )

    else:
        raise ValueError("Invalid dataset name")

    return trainset, testset


def get_loader(dataset, args, eval=False):
    # prepare sampler
    if not eval:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # prepare dataloader
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.train_batch_size if not eval else args.eval_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return loader
