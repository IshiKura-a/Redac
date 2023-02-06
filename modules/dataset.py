import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Union, List, Callable, Any, Tuple, Optional
from utils.logger import logger


class BaseDataset(Dataset):
    def __init__(self, root_dir: str, target_name: str):
        self.root_dir = root_dir
        self.target_name = target_name
        self.eval_transform = None
        self.train_transform = None
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2
        }
        self.split_array = None
        self.data_dir = None
        self.filename_array = None
        self.y_array = None

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, item):
        y = self.y_array[item]
        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[item])
        img = Image.open(img_filename).convert("RGB")
        # Figure out split and transform accordingly
        if self.split_array[item] == self.split_dict["train"] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[item] in [self.split_dict["val"], self.split_dict["test"]] and
              self.eval_transform):
            img = self.eval_transform(img)
        x = img
        return x, y

    def get_splits(self, splits: list, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in list(self.split_dict.keys()), split + " is not a valid split"
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
            setattr(subsets[split], "y_array", self.y_array[indices])
            subsets[split].group_array = self.group_array[indices]
        return subsets

    @property
    def n_train(self):
        return torch.count_nonzero(torch.Tensor(self.split_array) == self.split_dict["train"]).item()

    @property
    def n_val(self):
        return torch.count_nonzero(torch.Tensor(self.split_array) == self.split_dict["val"]).item()

    @property
    def n_test(self):
        return torch.count_nonzero(torch.Tensor(self.split_array) == self.split_dict["test"]).item()


class ConfounderDataset(BaseDataset):
    def __init__(self, root_dir: str, target_name: str, confounder_names: List[str]):
        super().__init__(root_dir, target_name)
        self.group_counts = None
        self.group_array = None
        self.confouder_names = confounder_names

    def __getitem__(self, item):
        g = self.group_array[item]
        x, y = super().__getitem__(item)

        return x, y, g

    def get_splits(self, splits: list, train_frac=1.0):
        subsets = super().get_splits(splits, train_frac)
        for split in splits:
            assert split in list(self.split_dict.keys()), split + " is not a valid split"
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            setattr(subsets[split], "group_array", self.group_array[indices])
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


class RedacDataset(ConfounderDataset):
    def __init__(self, root_dir: str, target_name: str, confounder_names: List[str], backbone: nn.Module, k: int,
                 epsilon: float, feature: str):
        super().__init__(root_dir, target_name, confounder_names)
        self.cluster_counts = None
        self.cluster_array = None
        from modules.redac import Redac
        self.redac = Redac(backbone, feature)

    def __getitem__(self, item):
        try:
            x, y, g = super().__getitem__(item)
            c = self.cluster_array[item] if self.cluster_array is not None else -1
        except IndexError:
            raise
        return x, y, g, c

    def get_splits(self, splits: list, train_frac=1.0):
        subsets = super().get_splits(splits, train_frac)
        for split in splits:
            assert split in list(self.split_dict.keys()), split + " is not a valid split"
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            setattr(subsets[split], "cluster_array", self.cluster_array[indices])
        return subsets


class CelebADataset(RedacDataset):
    def __init__(self, root_dir: str, target_name: str, confounder_names: List[str], backbone: nn.Module, k: int,
                 epsilon: float, feature: str, augment_data: bool, apply_cluster: bool = True):
        super().__init__(root_dir, target_name, confounder_names, backbone, k, epsilon, feature)
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data

        # Read in attributes
        logger.info("Loading attributes")
        self.attrs_df = pd.read_csv(
            os.path.join(root_dir, f'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, f'img_align_celeba')
        self.filename_array = self.attrs_df["image_id"].values
        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype("int")
        self.group_counts = (
            torch.arange(self.n_groups).unsqueeze(1) == torch.Tensor(self.group_array)).sum(
            1).float()

        # Read in train/val/test splits
        logger.info("Loading splits")
        self.split_df = pd.read_csv(
            os.path.join(root_dir, f'list_eval_partition.csv'))
        self.split_array = self.split_df["partition"].values

        from modules.preprocess import get_transform_celebA
        self.train_transform = get_transform_celebA(is_train=True, augment_data=augment_data)
        self.eval_transform = get_transform_celebA(is_train=False, augment_data=augment_data)

        self.n_clusters = k
        if apply_cluster:
            logger.info("Start clustering")
            train_idx = np.where(self.split_array == self.split_dict["train"])[0]
            val_idx = np.where(self.split_array == self.split_dict["val"])[0]
            self.cluster_idx, cluster_array = self.redac.cluster(
                Subset(self, train_idx),
                Subset(self, val_idx),
                k, epsilon)
            self.cluster_array = -1 * torch.ones(len(self)).long()
            self.cluster_array[train_idx] = cluster_array[:len(train_idx)]
            self.cluster_array[val_idx] = cluster_array[len(train_idx):len(cluster_array)]
        else:
            logger.info("Skip clustering, random assign groups")
            self.cluster_array = -1 * torch.ones(len(self)).long()
            non_test_idx = self.split_array != self.split_dict["test"]
            self.cluster_array[non_test_idx] = torch.randint(low=0, high=k, size=(np.count_nonzero(non_test_idx),))
        self.cluster_counts = (torch.arange(k).unsqueeze(1) == torch.Tensor(self.cluster_array)).sum(
            1).float()

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)


class WaterBirdDataset(RedacDataset):
    def __init__(self, root_dir: str, target_name: str, confounder_names: List[str], backbone: nn.Module, k: int,
                 epsilon: float, feature: str, augment_data: bool, apply_cluster: bool = True):
        super().__init__(root_dir, target_name, confounder_names, backbone, k, epsilon, feature)
        self.root_dir = root_dir
        self.data_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data

        if not os.path.exists(self.root_dir):
            raise ValueError(
                f'{self.root_dir} does not exist yet. Please generate the dataset first.')

        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, "metadata.csv"))

        # Get the y values
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype("int")
        self.group_counts = (
            torch.arange(self.n_groups).unsqueeze(1) == torch.Tensor(self.group_array)).sum(
            1).float()

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values

        from modules.preprocess import get_transform_waterbird
        self.train_transform = get_transform_waterbird(is_train=True, augment_data=augment_data)
        self.eval_transform = get_transform_waterbird(is_train=False, augment_data=augment_data)

        self.n_clusters = k
        if apply_cluster:
            logger.info("Start clustering")
            train_idx = np.where(self.split_array == self.split_dict["train"])[0]
            val_idx = np.where(self.split_array == self.split_dict["val"])[0]
            self.cluster_idx, cluster_array = self.redac.cluster(
                Subset(self, train_idx),
                Subset(self, val_idx),
                k, epsilon)
            self.cluster_array = -1 * torch.ones(len(self)).long()
            self.cluster_array[train_idx] = cluster_array[:len(train_idx)]
            self.cluster_array[val_idx] = cluster_array[len(train_idx):len(cluster_array)]
        else:
            logger.info("Skip clustering, random assign groups")
            self.cluster_array = -1 * torch.ones(len(self)).long()
            non_test_idx = self.split_array != self.split_dict["test"]
            self.cluster_array[non_test_idx] = torch.randint(low=0, high=k, size=(np.count_nonzero(non_test_idx),))
        self.cluster_counts = (torch.arange(k).unsqueeze(1) == torch.Tensor(self.cluster_array)).sum(
            1).float()


class SubsetWrapper(Dataset):
    def __init__(self, dataset: ConfounderDataset, process_item_fn: Optional[Callable], n_groups: int, n_classes: int,
                 group_str_fn: Callable):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn

        if (self.dataset.y_array is not None) and (self.dataset.group_array is not None):
            self.group_array = torch.LongTensor(self.dataset.group_array)
            self.y_array = torch.LongTensor(self.dataset.y_array)
        else:
            group_array = []
            y_array = []
            loader_kwargs = {"batch_size": 1024, "num_workers": 4, "pin_memory": False}
            batch_loader = DataLoader(dataset, **loader_kwargs)
            for batch in tqdm(batch_loader, total=len(batch_loader)):
                y = batch[1]
                g = batch[2]
                group_array.append(g)
                y_array.append(y)
            self.group_array = torch.concatenate(group_array)
            self.y_array = torch.concatenate(y_array)

        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self.group_array).sum(1).float()
        self.y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self.y_array).sum(1).float()

    def __getitem__(self, idx: int) -> Any:
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def class_counts(self) -> torch.Tensor:
        return self.y_counts

    @property
    def input_size(self) -> Tuple:
        for x, y, g in self:
            return x.size()


class RedacSubsetWrapper(SubsetWrapper):
    def __init__(self, dataset: RedacDataset, process_item_fn: Optional[Callable], n_groups: int, n_classes: int,
                 group_str_fn: Callable, n_clusters: int):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        self.n_clusters = n_clusters

        if not [x for x in (self.dataset.y_array, self.dataset.group_array, self.dataset.cluster_array) if x is None]:
            self.group_array = torch.LongTensor(self.dataset.group_array)
            self.y_array = torch.LongTensor(self.dataset.y_array)
            self.cluster_array = self.dataset.cluster_array
        else:
            group_array = []
            y_array = []
            cluster_array = []

            loader_kwargs = {"batch_size": 1024, "num_workers": 4, "pin_memory": False}
            batch_loader = tqdm(DataLoader(dataset, **loader_kwargs))

            for batch_idx, batch in enumerate(batch_loader):
                y = batch[1]
                g = batch[2]
                c = batch[3]
                group_array.append(g)
                y_array.append(y)
                cluster_array.append(c)

            self.group_array = torch.concatenate(group_array)
            self.y_array = torch.concatenate(y_array)
            self.cluster_array = torch.concatenate(cluster_array)

        self.group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self.group_array).sum(1).float()
        self.y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self.y_array).sum(1).float()
        self.cluster_counts = (torch.arange(self.n_clusters).unsqueeze(1) == self.cluster_array).sum(1).float()
