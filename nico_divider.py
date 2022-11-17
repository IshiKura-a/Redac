import os
import json
from os.path import join
import io
import numpy as np
import scipy

import scipy.misc
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor
import imageio
import lmdb
import torch
from multiprocessing import Pool
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg
import pickle


def pil_loader(imgname, crop=None):
    # maybe faster
    if crop is None:
        with open(imgname, "rb") as f:
            return Image.open(imgname).convert("RGB")
    else:
        box = (crop[0].item(), crop[1].item(), crop[2].item() + crop[0].item(), crop[3].item() + crop[1].item())
        with open(imgname, "rb") as f:
            img = Image.open(imgname).convert("RGB")
        im = img.crop(box)

        return im


colorjit = [transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2, 0.1)]
root_dir = f'/data/home/tangzihao/dataset/nico'


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomApply(colorjit),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


def test_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


def to_pil_image():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    to_img = transforms.ToPILImage()

    def f(img):
        oimg = img * std + mean
        return to_img(oimg)

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])


def default_transform(a):
    return ToTensor()(a)


def collate_fn_NICO():
    def f(data):
        path = []
        gt = []
        context = []
        img = []
        for d in tqdm(data):
            d_img = d["img"]
            d_img = torch.stack([d_img[0], d_img[1], d_img[2]], dim=-1)
            d_img = Image.fromarray(np.uint8(255 * d_img))

            path.append(d["path"])
            gt.append(d["gt"])
            context.append(d["context"])
            img.append(d_img)

        return {
            "path": path,
            "gt": torch.from_numpy(np.array(gt)).long(),
            "context": torch.from_numpy(np.array(context)).long(),
            "img": img
        }

    return f


class NICOAnimal_LmdbDataset(Dataset):
    context = ["white", "in circus", "running", "eating", "at home", "on ground", "on snow",
               "lying", "in cage", "sitting", "aside people", "in street", "black", "climbing",
               "eating grass", "in forest", "spotted", "in zoo", "on tree", "standing", "flying", "in water",
               "in hand", "on grass", "in hole", "at sunset", "walking", "on beach", "brown", "on shoulder",
               "in river", "on branch", "on road"]
    category = ["bear", "bird", "cat", "cow", "dog", "elephant", "horse", "monkey", "rat", "sheep"]

    def __init__(self, root, transform=default_transform):
        super().__init__()
        self.root = root
        self.env = None
        self.txn = None
        self.transform = transform
        if self.env is None:
            self._init_db()

    def _init_db(self):
        self.env = lmdb.open(self.root,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()
        image_bin = self.txn.get(f"{index}".encode("utf-8"))
        if image_bin is None:
            raise IndexError("Index out of range")
        (img, path, ctgy, ctx) = pickle.loads(image_bin)
        img = Image.open(io.BytesIO(img))

        return {
            "path": path,
            "img": self.transform(img),
            "gt": self.category.index(ctgy),
            "context": self.context.index(ctx),
        }

    def __len__(self):
        if self.env is None:
            self._init_db()
        return self.env.stat()["entries"]


def main():
    data = NICOAnimal_LmdbDataset(f'{root_dir}/nico_animal_prop_D4_DR3_S256_train.lmdb')

    for pt in data.category:
        for ctx in data.context:
            Path(f'{root_dir}/train/{pt}/{ctx}').mkdir(parents=True, exist_ok=True)

    for d in tqdm(data):
        _, gt, ctx, file_name = str(d["path"]).split('\\')
        tmp = np.uint8(255 * torch.stack(list(d["img"]), dim=-1))
        Image.fromarray(tmp).save(f'{root_dir}/train/{gt}/{ctx}/{file_name}')


if __name__ == '__main__':
    main()
