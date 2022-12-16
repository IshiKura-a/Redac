import torch
import numpy as np

from collections import Counter
from sklearn.cluster import KMeans
from utils.logger import log_kmeans_cluster


def baseline_kmeans(best_k, features, dataset):
    kmeans = KMeans(n_clusters=best_k).fit(features.flatten(start_dim=1, end_dim=len(features.shape) - 1))
    info = []
    for i in range(best_k):
        member = (torch.Tensor(kmeans.labels_) == i).nonzero(as_tuple=False)
        contexts = [dataset["label"][x] for x in member]
        counter = Counter(contexts)
        info.append([counter.most_common(1)[0][0], counter.most_common(1)[0][1], len(member),
                     counter.most_common(1)[0][1] / len(member)])

    log_kmeans_cluster(info)
