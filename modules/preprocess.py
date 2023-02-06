import torch
import numpy as np

from torch import Tensor, LongTensor
from typing import Optional, Union
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import InterpolationMode

from modules.dataset import BaseDataset, SubsetWrapper, RedacSubsetWrapper
from utils.logger import logger


def get_transform_celebA(is_train: bool, augment_data: bool = False) -> transforms.Compose:
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    if (not is_train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction anymore
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_transform_waterbird(is_train: bool, augment_data: bool = False) -> transforms.Compose:
    scale = 256.0/224.0
    target_resolution = (224, 224)
    assert target_resolution is not None

    if (not is_train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(dataset: Union[Subset, BaseDataset, SubsetWrapper, RedacSubsetWrapper], is_train: bool,
               reweight_groups: Optional[bool] = None,
               reweight_clusters: Optional[bool] = None, **kwargs) -> DataLoader:
    # Two methods cannot be applied simultaneously
    assert None in (reweight_clusters, reweight_groups)
    if not is_train:
        assert reweight_groups is None and reweight_clusters is None
        shuffle = False
        sampler = None
    elif not reweight_groups and not reweight_clusters:
        shuffle = True
        sampler = None
    elif reweight_groups:
        group_weights = len(dataset) / dataset.group_counts
        weights = group_weights[dataset.group_array]

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        shuffle = False
    else:
        cluster_weights = len(dataset) / dataset.cluster_counts
        weights = cluster_weights[dataset.cluster_array]

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        shuffle = False

    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
