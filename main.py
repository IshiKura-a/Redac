import json
import sys
import argparse
import torch
import numpy as np

from typing import Optional
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, Dataset

from modules.redac import Redac
from utils.baseline import baseline_kmeans
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, default="microsoft/resnet-50")
    # default to f'/data/home/tangzihao/dataset/nico/train/bear/'
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--k", type=int, default=21)

    args = parser.parse_args()
    # dataset = load_dataset("imagefolder", data_dir=args.input_dir)
    # dataset = dataset["train"]
    #
    # feature_extractor = AutoFeatureExtractor.from_pretrained(args.backbone)
    # model = ResNetForImageClassification.from_pretrained(args.backbone)
    # front_end = Redac(model, feature_extractor)
    # cluster_index, assignment = front_end.cluster(dataset, args.k)

    # log_redac_cluster(args.k, cluster_index, assignment, front_end.probs, dataset, model.id2label)
    # baseline_kmeans(args.k, front_end.features, dataset)

    dataset = load_dataset("csv", data_dir=args.input_dir)


if __name__ == '__main__':
    main()
