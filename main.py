import json
import sys
import os
import argparse
import torch
import torchvision
import numpy as np

from typing import Optional
from torch import nn
from torchvision.models import ResNet50_Weights

from modules.dataset import CelebADataset, SubsetWrapper, RedacSubsetWrapper
from modules.preprocess import get_transform_celebA, get_loader, set_seed
from modules.redac import Redac
from modules.train import train
from utils.baseline import baseline_kmeans
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, default="resnet-50")
    # default to f'/data/home/tangzihao/dataset/nico/train/bear/'
    parser.add_argument("--root_dir", type=str, default=f'/data/home/tangzihao/dataset/CelebA')
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--log_dir", type=str, default=f'/data/home/tangzihao/model/group_dro_resnet')

    # Optimization
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--robust_step_size", default=0.01, type=float)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)

    args = parser.parse_args()

    set_seed(args.seed)

    logger.info("Loading model")
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    full_dataset = CelebADataset(
        root_dir=args.root_dir,
        target_name="Blond_Hair",
        confounder_names=["Male"],
        augment_data=False,
        backbone=model,
        k=10
    )

    logger.info("Splitting datasets")
    split = full_dataset.get_splits(["train", "val", "test"])
    train_data, val_data, test_data = [
        RedacSubsetWrapper(split[i], process_item_fn=None, n_groups=full_dataset.n_groups,
                           n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str,
                           n_clusters=full_dataset.n_clusters) for i in ["train", "val", "test"]
    ]

    d = model.fc.in_features
    model.fc = nn.Linear(d, full_dataset.n_classes)

    logger.info("Getting data loaders")
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True}
    train_loader = get_loader(train_data, is_train=True, reweight_clusters=True, **loader_kwargs)
    val_loader = get_loader(val_data, is_train=False, **loader_kwargs)
    test_loader = get_loader(test_data, is_train=False, **loader_kwargs) if test_data is not None else None

    data = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
            "train_data": train_data, "val_data": val_data, "test_data": test_data}

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    train(model, criterion, data, args, epoch_offset=0)


if __name__ == "__main__":
    main()
