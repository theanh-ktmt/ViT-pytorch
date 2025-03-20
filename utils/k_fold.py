import numpy as np
from sklearn.model_selection import StratifiedKFold

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Subset
from utils.data_utils import get_loader, NUM_CLASS_MAPPING
from models.modeling import VisionTransformer, CONFIGS


class KFoldDataset(torch.utils.data.Dataset):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.n_splits = args.k_fold
        self.splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=args.seed
        )
        self.folds = []

    def split(self, dataset):
        self.folds = []

        # prepare inputs
        labels = np.array(dataset.targets)
        fake_inputs = np.zeros(len(labels))

        fold_iterator = self.splitter.split(fake_inputs, labels)
        for train_idx, val_idx in fold_iterator:
            # create subset
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # create dataloader
            train_loader = get_loader(train_subset, self.args)
            val_loader = get_loader(val_subset, self.args, eval=True)

            # add to fold list
            self.folds.append((train_loader, val_loader))

        return self

    def get_fold(self, fold_idx):
        return self.folds[fold_idx]


class KFoldEnsembleModel(nn.Module):
    def __init__(self, model_paths, args: argparse.Namespace, decide_mode="mean"):
        super().__init__()
        self.args = args
        self.model_paths = model_paths
        self.decide_mode = decide_mode
        self.models = [self.load_model(model_path) for model_path in model_paths]
        self.n_models = len(self.models)

    def load_model(self, model_path: str):
        config = CONFIGS[self.args.model_type]
        num_classes = NUM_CLASS_MAPPING[self.args.dataset]
        model = VisionTransformer(
            config, self.args.img_size, zero_head=True, num_classes=num_classes
        )
        model.load_from(np.load(model_path))
        return model

    def forward(self, x):
        logits = [model(x)[0] for model in self.models]
        if self.decide_mode == "mean":
            return torch.stack(logits).mean(dim=0)
        elif self.decide_mode == "max":
            return torch.stack(logits).max(dim=0)
        else:
            raise ValueError("Invalid decide mode!")
