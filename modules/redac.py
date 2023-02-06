import sys
import random
from typing import Optional, List, Tuple, Generic, Union, TypeVar

import torch
from torch import nn, Tensor
from torch.masked import masked_tensor
from torch.nn.functional import softmax
from torch.utils.data import Subset, DataLoader
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from tqdm import tqdm

from utils.logger import logger


class Redac:
    """
    - Description: Basic Version of Redac front end
    - Require: Backbone model and its extractor, Resnet-50 is commonly used
    - Mind: The backbone model must give a dict-like object with "pixel_values" as one of the keys
    - Usage: Redac(model, extractor).cluster(dataset, k) --> cluster_index, assignment
    """

    def __init__(self, model: nn.Module, feature: str) -> None:
        self.model = model.cuda()
        self.feature = None

        def get_features(module, inputs, outputs):
            self.feature = outputs

        getattr(self.model, feature).register_forward_hook(get_features)
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
            sim = pairwise_cosine_similarity(flattened_x, flattened_y)
            return sim, 1 - sim
        elif mode == "l2":
            dist = pairwise_euclidean_distance(flattened_x, flattened_y)
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

    def cluster(self, train_dataset: Subset, val_dataset: Subset, k: int, epsilon: float) -> Tuple[Tensor, Tensor]:
        loader_kwargs = {"batch_size": 128, "num_workers": 4, "pin_memory": True}
        batch_loader = DataLoader(train_dataset, **loader_kwargs)
        self.model.eval()

        logger.info("Use backbone to get probabilities")
        probs = []
        labels = []
        cluster_feature = []
        with torch.no_grad():
            for batch in tqdm(batch_loader, total=len(batch_loader)):
                batch = tuple(t.cuda() for t in batch)
                x = batch[0]
                y = batch[1]
                outputs = self.model(x)
                batch_probs = softmax(outputs, dim=1)
                probs.append(batch_probs)
                labels.append(y)
                self.feature = self.feature[batch_probs.max(dim=1)[0] > epsilon]
                if self.feature.shape[0] > 0:
                    cluster_feature.append(self.feature)

        probs = torch.concatenate(probs).cpu()
        self._probs = probs
        labels = torch.concatenate(labels).cpu()

        label_ids, label_counts = torch.unique(labels, return_counts=True, sorted=True)
        probs_max = probs.max(dim=1)[0]
        cluster_candidate = (probs_max > epsilon).nonzero(as_tuple=False).squeeze().long()

        label_in_candidates, counts = torch.unique(labels[cluster_candidate], return_counts=True, sorted=True)
        lack = (counts < k // len(label_ids)).nonzero(as_tuple=False).squeeze().shape[0] > 0
        not_enough = (counts < int(k // len(label_ids) * 1.5)).nonzero(as_tuple=False).squeeze().shape[0] > 0

        logger.info(f'Get {len(cluster_candidate)} cluster candidates for {label_in_candidates}: {counts}')
        if k % len(label_ids) != 0:
            logger.warning(f'k={k} is not a multiple of #label={len(label_ids)}, assign clusters randomly')

        def decomposition(m, n):
            for i in range(n - 1):
                out = random.randint(0, m)
                yield out
                m -= out
            yield m

        num_clusters = list(decomposition(k % len(label_ids), len(label_ids)))
        num_clusters = (torch.Tensor(num_clusters) + k // len(label_ids)).int()
        logger.info(f'number of clusters per label: {num_clusters}')

        if not_enough or counts.shape[0] != len(label_ids):
            logger.warning(f'Too small epsilon {epsilon}, please change to a larger one')
            if lack or counts.shape[0] != len(label_ids):
                if counts.shape[0] != len(label_ids):
                    logger.critical(f'{len(label_ids) - counts.shape[0]} labels does not appear in candidates')
                rem = 1 - epsilon
                logger.critical(f'Cluster candidates are not enough, system crashed!')
                logger.critical(f'Infos of other epsilons:')
                logger.critical(
                    f'epsilon {1 - (rem * 10):.6f}: {torch.count_nonzero(probs_max > 1 - (rem * 10))} candidates\n'
                    f'epsilon {1 - (rem * 20):.6f}: {torch.count_nonzero(probs_max > 1 - (rem * 20))} candidates\n'
                    f'epsilon {1 - (rem * 50):.6f}: {torch.count_nonzero(probs_max > 1 - (rem * 50))} candidates\n'
                    f'epsilon {1 - (rem * 100):.6f}: {torch.count_nonzero(probs_max > 1 - (rem * 100))} candidates\n'
                )
                sys.exit(-1)

        logger.info("Start to calculate similarities")
        cluster_feature = torch.concatenate(cluster_feature)
        similarity, distance = self.discriminate(cluster_feature.cuda(), mode="l2")

        cluster_index_in_candidate = []
        cluster_mask = torch.zeros((len(label_ids), k))
        accu = 0
        for (label_id, num) in zip(label_ids.tolist(), num_clusters.tolist()):
            idx = (labels[cluster_candidate] == label_id).nonzero(as_tuple=False).squeeze().long().cuda()
            cluster_dist = distance[idx, :][:, idx].sum(dim=0)
            cluster_index_in_candidate.append(cluster_dist.topk(num).indices.cpu())
            cluster_mask[label_id, :][accu:accu + num] = 1
            accu += num
        cluster_mask = cluster_mask.bool().cuda()

        cluster_index_in_candidate = torch.concatenate(cluster_index_in_candidate)
        cluster_index = cluster_candidate[cluster_index_in_candidate].long()
        cluster_feature = cluster_feature[cluster_index_in_candidate]

        logger.info("Starting assignments")
        assignment = []
        for batch in tqdm(batch_loader, total=len(batch_loader)):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            _ = self.model(x)
            sim, _ = self.discriminate(self.feature, cluster_feature.cuda(), mode="l2")
            mask = cluster_mask[y]
            ass = masked_tensor(sim.clone().detach(), mask).argmax(dim=-1)
            assignment.append(ass.to_tensor(0).squeeze())

        logger.info("Starting to cluster val data")
        batch_loader = DataLoader(val_dataset, **loader_kwargs)
        for batch in tqdm(batch_loader, total=len(batch_loader)):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            _ = self.model(x)
            sim, _ = self.discriminate(self.feature, cluster_feature.cuda(), mode="l2")
            mask = cluster_mask[y]
            ass = masked_tensor(sim.clone().detach(), mask).argmax(dim=-1)
            assignment.append(ass.to_tensor(0).squeeze())

        assignment = torch.concatenate(assignment).cpu().long()
        self.log_info(k, assignment)
        return cluster_index, assignment

    @property
    def probs(self):
        return self.probs
