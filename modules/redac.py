import torch

from typing import Any, Optional, List
from torch import nn, Tensor
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from datasets import Dataset
from torch.nn.functional import softmax
from utils.logger import logger


class Redac:
    """
    - Description: Basic Version of Redac front end
    - Require: Backbone model and its extractor, Resnet-50 is commonly used
    - Mind: The backbone model must give a dict-like object with "pixel_values" as one of the keys
    - Usage: Redac(model, extractor).cluster(dataset, k) --> cluster_index, assignment
    """

    def __init__(self, model: nn.Module, extractor: Any):
        self.model = model
        self.extractor = extractor
        self._features = None
        self._probs = None

    @staticmethod
    def flattened_cosine_similarity(x: Tensor, y: Tensor):
        flattened_x = x.flatten(start_dim=1, end_dim=len(x.shape) - 1)
        if y is None:
            return pairwise_cosine_similarity(flattened_x, zero_diagonal=False)
        else:
            flattened_y = y.flatten(start_dim=1, end_dim=len(y.shape) - 1)
            return pairwise_cosine_similarity(flattened_x, flattened_y, zero_diagonal=False)

    @staticmethod
    def discriminate(x: Tensor, y: Optional[Tensor] = None, mode: Optional[str] = "cosine"):
        flattened_x = x.flatten(start_dim=1, end_dim=len(x.shape) - 1)
        flattened_y = flattened_x if y is None else y.flatten(start_dim=1, end_dim=len(y.shape) - 1)
        if mode == "cosine":
            sim = pairwise_cosine_similarity(flattened_x, flattened_y, zero_diagonal=False)
            return sim, 1 - sim
        elif mode == "l2":
            dist = pairwise_euclidean_distance(flattened_x, flattened_y, zero_diagonal=True)
            max_dist = torch.max(dist)
            return (max_dist - dist) / max_dist, dist

    @staticmethod
    def log_info(k: int, assignment: Tensor):
        logger.info(f'Using {k} clusters')
        logger.info("------------Cluster Info-------------")
        info: List = []
        for i in range(k):
            member = (assignment == i).nonzero(as_tuple=False).squeeze().long()
            if not isinstance(member.tolist(), list):
                member = [member]
            logger.info(f'{k}th cluster has {len(member)} members.')
            info.append(len(member))

        info: Tensor = Tensor(info)
        logger.info(f'MAX: {info.max():3} MIN: {info.min():3} AVG: {info.mean():5.2} VAR: {info.var():7.2}')

    """
    Description: Key method of Redac front end, solving the Random Covering problem by minimizing distance
    Require: dataset: to cluster, k: cluster number, epsilon: representative data threshold
    Mind: dataset is a Dataset object, which has "image" as a key
    """

    def cluster(self, dataset: Dataset, k: int, epsilon: float = 0.99999):
        inputs = self.extractor(dataset["image"], return_tensors="pt")
        features = inputs.data["pixel_values"]
        self._features = features
        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits

        probs = softmax(logits, dim=1)
        self._probs = probs
        probs_max = probs.max(dim=1)[0].sort(descending=True)
        cluster_candidate = probs_max.indices[probs_max.values > epsilon]

        similarity, distance = self.discriminate(features, mode="l2")
        cluster_dist = distance[cluster_candidate, :][:, cluster_candidate].sum(dim=0)

        cluster_index = cluster_candidate[cluster_dist.topk(k).indices]
        filtered_similarity = similarity[:, cluster_index]
        assignment = filtered_similarity.argmax(dim=-1)
        self.log_info(k, assignment)

        return cluster_index, assignment

    @property
    def features(self):
        return self._features

    @property
    def probs(self):
        return self.probs
