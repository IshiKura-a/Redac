import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from abc import abstractmethod
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
        self._y_counts = None

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, item):
        y = self.y_array[item]
        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[item])
        img = Image.open(img_filename).convert('RGB')
        # Figure out split and transform accordingly
        if self.split_array[item] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[item] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)
        x = img
        return x, y

    def get_splits(self, splits: list, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in list(self.split_dict.keys()), split + ' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    @property
    def y_counts(self):
        if self._y_counts is None:
            self._y_counts = (
                    torch.arange(self.n_classes).unsqueeze(1) == torch.Tensor(self.y_array)).sum(
                1).float()
        return self._y_counts

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
        self._group_counts = None
        self.group_array = None
        self.confouder_names = confounder_names

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, item):
        g = self.group_array[item]
        x, y = super().__getitem__(item)

        return x, y, g

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name

    @property
    def group_counts(self):
        return self._group_counts


class RedacDataset(ConfounderDataset):
    def __init__(self, root_dir: str, target_name: str, confounder_names: List[str], backbone: nn.Module, k: int):
        super().__init__(root_dir, target_name, confounder_names)
        self._cluster_counts = None
        self._cluster_array = None
        from modules.redac import Redac
        self.redac = Redac(backbone)

    def __getitem__(self, item):
        x, y, g = super().__getitem__(item)
        c = self._cluster_array[item] if self._cluster_array is not None else -1
        return x, y, g, c

    @property
    def cluster_array(self):
        return self._cluster_array

    @property
    def cluster_counts(self):
        return self._cluster_counts


class CelebADataset(RedacDataset):
    def __init__(self, root_dir: str, target_name: str, confounder_names: List[str], backbone: nn.Module, k: int,
                 augment_data: bool):
        super().__init__(root_dir, target_name, confounder_names, backbone, k)
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
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype('int')
        self._group_counts = (
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

        logger.info("Start clustering")
        self.n_clusters = k
        self.cluster_idx, self._cluster_array = self.redac.cluster(Subset(self, np.where(self.split_array == 0)[0]), k)
        # self.cluster_idx, self._cluster_array = torch.arange(10), torch.randint(0, 10, size=(self.n_train,))
        self._cluster_array = torch.concatenate(
            [self._cluster_array, -1 * torch.ones(self.n_val + self.n_test)]).long()
        self._cluster_counts = (torch.arange(k).unsqueeze(1) == torch.Tensor(self._cluster_array)).sum(
            1).float()

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)


class SubsetWrapper(Dataset):
    def __init__(self, dataset: ConfounderDataset, process_item_fn: Optional[Callable], n_groups: int, n_classes: int,
                 group_str_fn: Callable):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        loader_kwargs = {"batch_size": 1024, "num_workers": 4, "pin_memory": False}
        batch_loader = DataLoader(dataset, **loader_kwargs)
        for batch_idx, batch in enumerate(batch_loader):
            y = batch[1]
            g = batch[2]
            group_array.append(g)
            y_array.append(y)

        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)== self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()

    def __getitem__(self, idx: int) -> Any:
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def class_counts(self) -> torch.Tensor:
        return self._y_counts

    @property
    def input_size(self) -> Tuple:
        for x, y, g in self:
            return x.size()

    @property
    def group_counts(self) -> torch.Tensor:
        return self._group_counts

    @property
    def group_array(self) -> torch.Tensor:
        return self._group_array


class RedacSubsetWrapper(SubsetWrapper):
    def __init__(self, dataset: RedacDataset, process_item_fn: Optional[Callable], n_groups: int, n_classes: int,
                 group_str_fn: Callable, n_clusters: int):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        self.n_clusters = n_clusters

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

        self._group_array = torch.concatenate(group_array)
        self._y_array = torch.concatenate(y_array)
        self._cluster_array = torch.concatenate(cluster_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()
        self._cluster_counts = (torch.arange(self.n_clusters).unsqueeze(1) == self._cluster_array).sum(1).float()

    @property
    def cluster_array(self) -> torch.Tensor:
        return self._cluster_array

    @property
    def cluster_counts(self) -> torch.Tensor:
        return self._cluster_counts
