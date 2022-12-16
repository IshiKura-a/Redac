import json
import multiprocessing
import sys

import numpy as np
import torch
from collections import Counter
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, Image, Dataset
from matplotlib import pyplot as plt
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
from main import preprosess, discriminate, log_redac_cluster, loss_func


def main():
    dataset, features, probs, id2label = preprosess()
    probs_max = probs.max(dim=1)[0].sort(descending=True)
    cluster_candidate = probs_max.indices[probs_max.values > 0.99999]
    similarity, distance = discriminate(features, mode="l2")
    cluster_dist = distance[cluster_candidate, :][:, cluster_candidate].sum(dim=0)

    k = 24
    flattened_features = features.flatten(start_dim=1, end_dim=len(features.shape) - 1)
    cluster_index = cluster_candidate[cluster_dist.topk(k).indices]
    kmeans = KMeans(n_clusters=k, init=flattened_features[cluster_index]).fit(flattened_features)
    cur_loss = kmeans.inertia_
    assignment = torch.Tensor(kmeans.labels_)
    group_num = torch.Tensor([assignment.tolist().count(_i) for _i in range(k)])
    cluster_similarity = similarity[cluster_index, :][:, cluster_index]
    norm = torch.linalg.norm(cluster_similarity)
    penalty = len((group_num <= 2).nonzero(as_tuple=False))
    min_loss = loss_func(cur_loss, norm, penalty)

    print(f'#Cluster: {k:3} Loss: {min_loss:7.6} with {cur_loss:7.6} Norm: {norm:7.6} Penalty: {penalty:3}')

    log_redac_cluster(k, cluster_index, assignment, similarity, min_loss, probs, dataset, id2label)


if __name__ == '__main__':
    main()
