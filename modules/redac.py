import sys

import torch

from typing import Any, Optional, List, Tuple
from torch import nn, Tensor
from torch.utils.data import Dataset, Subset, DataLoader
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from torch.nn.functional import softmax
from tqdm import tqdm

from modules.dataset import BaseDataset, RedacDataset
from utils.logger import logger


class Redac:
    """
    - Description: Basic Version of Redac front end
    - Require: Backbone model and its extractor, Resnet-50 is commonly used
    - Mind: The backbone model must give a dict-like object with "pixel_values" as one of the keys
    - Usage: Redac(model, extractor).cluster(dataset, k) --> cluster_index, assignment
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model.cuda()
        self._probs = None

    @staticmethod
    def flattened_cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
        flattened_x = x.flatten(start_dim=1, end_dim=len(x.shape) - 1)
        if y is None:
            return pairwise_cosine_similarity(flattened_x, zero_diagonal=False)
        else:
            flattened_y = y.flatten(start_dim=1, end_dim=len(y.shape) - 1)
            return pairwise_cosine_similarity(flattened_x, flattened_y, zero_diagonal=False)

    @staticmethod
    def discriminate(x: Tensor, y: Optional[Tensor] = None, mode: Optional[str] = "cosine") -> Tuple[Tensor, Tensor]:
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
    def log_info(k: int, assignment: Tensor) -> None:
        logger.info(f'Using {k} clusters')
        logger.info("------------Cluster Info-------------")
        info: List = []
        for i in range(k):
            member = (assignment == i).nonzero(as_tuple=False).squeeze().long()
            if not isinstance(member.tolist(), list):
                member = [member]
            logger.info(f'{i}th cluster has {len(member)} members.')
            info.append(len(member))

        info: Tensor = Tensor(info)
        logger.info(f'MAX: {info.max():3} MIN: {info.min():3} AVG: {info.mean():5.2} VAR: {info.var():7.2}')

    """
    Description: Key method of Redac front end, solving the Random Covering problem by minimizing distance
    Require: dataset: to cluster, k: cluster number, epsilon: representative data threshold
    Mind: dataset is a Dataset object, which has "image" as a key
    """

    def cluster(self, dataset: Subset[RedacDataset], k: int, epsilon: float = 0.9997) -> Tuple[Tensor, Tensor]:
        from modules.preprocess import get_loader
        loader_kwargs = {"batch_size": 512, "num_workers": 4, "pin_memory": False}
        batch_loader = DataLoader(dataset, **loader_kwargs)
        self.model.eval()

        logger.info("Use backbone to get probabilities")
        probs = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(batch_loader)):
                batch = tuple(t.cuda() for t in batch)
                x = batch[0]
                outputs = self.model(x)
                batch_probs = softmax(outputs, dim=1)
                probs.append(batch_probs)

        self._probs = probs = torch.concatenate(probs).cpu()
        probs_max = probs.max(dim=1)[0]
        cluster_candidate = (probs_max > epsilon).nonzero(as_tuple=False).squeeze().long()

        logger.info(f'Get {len(cluster_candidate)} cluster candidates')
        if len(cluster_candidate) < k * 2:
            logger.warning(f'Too small epsilon {epsilon}, please change to a larger one')
            if len(cluster_candidate) < k:
                rem = 1 - epsilon
                logger.critical(f'Cluster candidate is not enough, system crashed!')
                logger.critical(f'Infos of other epsilons:')
                logger.critical(
                    f'epsilon {1 - (rem * 10)}: {torch.count_nonzero(probs_max > 1 - (rem * 10))} candidates\n'
                    f'epsilon {1 - (rem * 20)}: {torch.count_nonzero(probs_max > 1 - (rem * 20))} candidates\n'
                    f'epsilon {1 - (rem * 50)}: {torch.count_nonzero(probs_max > 1 - (rem * 50))} candidates\n'
                    f'epsilon {1 - (rem * 100)}: {torch.count_nonzero(probs_max > 1 - (rem * 100))} candidates\n'
                    f'epsilon {1 - (rem * 1000)}: {torch.count_nonzero(probs_max > 1 - (rem * 1000))} candidates\n'
                )
                sys.exit(-1)

        logger.info("Start to calculate similarities")
        cluster_feature = torch.stack([dataset[i][0] for i in cluster_candidate])
        similarity, distance = self.discriminate(cluster_feature.cuda(), mode="l2")
        cluster_dist = distance.sum(dim=0)

        cluster_index_in_candidate = cluster_dist.topk(k).indices.cpu()
        cluster_index = cluster_candidate[cluster_index_in_candidate].long()
        cluster_feature = cluster_feature[cluster_index_in_candidate]

        logger.info("Starting assignments")
        assignment = torch.Tensor().cuda()
        for batch_idx, batch in tqdm(enumerate(batch_loader)):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            sim, _ = self.discriminate(x, cluster_feature.cuda(), mode="l2")
            ass = sim.argmax(dim=-1)
            assignment = torch.concatenate((assignment, ass))

        assignment = assignment.cpu().long()
        self.log_info(k, assignment)
        return cluster_index, assignment

    @property
    def features(self):
        return self._features

    @property
    def probs(self):
        return self.probs
