import json
import logging
import sys
import os
import argparse
import torch
import torchvision
import numpy as np

from typing import Optional
from datetime import datetime
from torch import nn
from torchvision.models import ResNet50_Weights, ResNet
from pathlib import Path
from modules.dataset import CelebADataset, SubsetWrapper, RedacSubsetWrapper, WaterBirdDataset
from modules.preprocess import get_transform_celebA, get_loader, set_seed
from modules.train import train
from utils.logger import logger, print_args


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=Optional[str],
                        default=None)
    parser.add_argument("--dataset", type=str, default="WaterBird")
    parser.add_argument("--root_dir", type=str, default=f'/data/home/tangzihao/dataset/waterbird_complete95_forest2water2')
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--e", type=float, default=0.999)
    parser.add_argument("--feature", type=str, default="layer1")
    parser.add_argument("--log_dir", type=str, default=f'/data/home/tangzihao/model/redac/wbs')

    # Optimization
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--robust_step_size", default=0.01, type=float)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--save_last", action="store_true", default=True)

    # Ablation
    parser.add_argument("--random_groups", action="store_true", default=False)

    # Baseline
    parser.add_argument("--baseline", choices=["gdro"], default=None)

    args = parser.parse_args()
    feature = ("rg" if args.random_groups else args.feature)
    postfix = ("" if args.baseline is None else args.baseline)
    postfix = "_" + postfix if len(postfix) > 0 else postfix

    args.log_dir = os.path.join(args.log_dir,
                                f'k{args.k}_e{args.e}_w{args.weight_decay}_lr{args.lr}_{feature}{postfix}_{datetime.now().strftime("%m%d")}')

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    logger.addHandler(logging.FileHandler(f'{args.log_dir}/result.log', mode="w"))
    set_seed(args.seed)

    print_args(args)
    logger.info("Loading model")

    if args.backbone is not None:
        model = torch.load(args.backbone)
    else:
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    if args.dataset == "CelebA":
        full_dataset = CelebADataset(
            root_dir=args.root_dir,
            target_name="Blond_Hair",
            confounder_names=["Male"],
            augment_data=False,
            backbone=model,
            k=args.k,
            epsilon=args.e,
            feature=args.feature,
            apply_cluster=(not args.random_groups) and (args.baseline is None)
        )
    elif args.dataset == "WaterBird":
        full_dataset = WaterBirdDataset(
            root_dir=args.root_dir,
            target_name="waterbird_complete95",
            confounder_names=["forest2water2"],
            augment_data=False,
            backbone=model,
            k=args.k,
            epsilon=args.e,
            feature=args.feature,
            apply_cluster=(not args.random_groups) and (args.baseline is None)
        )

    logger.info("Splitting datasets")
    split = full_dataset.get_splits(["train", "val", "test"])
    train_data, val_data, test_data = [
        RedacSubsetWrapper(split[i], process_item_fn=None, n_groups=full_dataset.n_groups,
                           n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str,
                           n_clusters=full_dataset.n_clusters) for i in ["train", "val", "test"]
    ]

    logger.info(f'Train size: {len(train_data)}')
    logger.info(f'Val size: {len(val_data)}')
    logger.info(f'Test size: {len(test_data)}')

    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    d = model.fc.in_features
    model.fc = nn.Linear(d, full_dataset.n_classes)

    logger.info("Getting data loaders")
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": True}
    if args.baseline is not None:
        train_loader = get_loader(train_data, is_train=True, reweight_groups=True, **loader_kwargs)
    else:
        train_loader = get_loader(train_data, is_train=True, reweight_clusters=True, **loader_kwargs)
    val_loader = get_loader(val_data, is_train=False, **loader_kwargs)
    test_loader = get_loader(test_data, is_train=False, **loader_kwargs) if test_data is not None else None

    data = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
            "train_data": train_data, "val_data": val_data, "test_data": test_data}

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    train(model, criterion, data, args, epoch_offset=0, baseline=args.baseline)


if __name__ == "__main__":
    main()
