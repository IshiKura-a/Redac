import json
import sys

import torch
from collections import Counter
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, Image, Dataset
from matplotlib import pyplot as plt
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from torch.nn.functional import softmax
from sklearn.cluster import KMeans


def flattened_cosine_similarity(x, y):
    flattened_x = x.flatten(start_dim=1, end_dim=len(x.shape) - 1)
    if y is None:
        return pairwise_cosine_similarity(flattened_x, zero_diagonal=False)
    else:
        flattened_y = y.flatten(start_dim=1, end_dim=len(y.shape) - 1)
        return pairwise_cosine_similarity(flattened_x, flattened_y, zero_diagonal=False)


def discriminate(x, y=None, mode="cosine"):
    flattened_x = x.flatten(start_dim=1, end_dim=len(x.shape) - 1)
    flattened_y = flattened_x if y is None else y.flatten(start_dim=1, end_dim=len(y.shape) - 1)
    if mode == "cosine":
        sim = pairwise_cosine_similarity(flattened_x, flattened_y, zero_diagonal=False)
        return sim, 1 - sim
    elif mode == "l2":
        dist = pairwise_euclidean_distance(flattened_x, flattened_y, zero_diagonal=True)
        max_dist = torch.max(dist)
        return (max_dist - dist) / max_dist, dist


dataset = load_dataset("imagefolder", data_dir=f'/data/home/tangzihao/dataset/nico/train/bear/')
dataset = dataset["train"]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = feature_extractor(dataset["image"], return_tensors="pt")
features = inputs.data["pixel_values"]
with torch.no_grad():
    output = model(**inputs)
    logits = output.logits

probs = softmax(logits, dim=1)
probs_max = probs.max(dim=1)[0].sort(descending=True)
cluster_candidate = probs_max.indices[probs_max.values > 0.99999]
# poor_image_index = probs_max.indices[probs_max.values < 0.90].tolist()
#
# with open(f'./output/poor_image.txt', 'w') as f:
#     for idx in poor_image_index:
#         print(dataset["train"]["name"][idx], file=f)
# print("Dump poor images")

similarity, distance = discriminate(features, mode="l2")
cluster_dist = distance[cluster_candidate, :][:, cluster_candidate].sum(dim=0)

min_loss = 1e30
best_k = 3
best_cluster_index = None
best_assignment = None
for k in range(3, len(cluster_dist)):
    cluster_index = cluster_candidate[cluster_dist.topk(k).indices]
    filtered_similarity = similarity[:, cluster_index]
    assignment = filtered_similarity.argmax(dim=-1)
    cur_loss = 0
    group_num = []
    for i in range(k):
        group_i_index = (assignment == i).nonzero(as_tuple=False)
        group_num.append(len(group_i_index))
        item_loss = distance[group_i_index, i]
        cur_loss += item_loss.mean().item()

    cur_loss = cur_loss / k
    group_num = torch.Tensor(group_num)
    cluster_similarity = similarity[cluster_index, :][:, cluster_index]
    norm = torch.linalg.norm(cluster_similarity) / len(cluster_index) * 100
    penalty = len((group_num <= 2).nonzero(as_tuple=False))
    alpha = 0.1
    beta = 0.01
    loss = (1 - alpha - beta) * cur_loss + alpha * norm + beta * penalty

    print(f'#Cluster: {k:3} Loss: {loss:7.6} with {cur_loss:7.6} Norm: {norm:7.6} Penalty: {penalty:3}')
    if loss < min_loss:
        # print(f'#Cluster: {k:3} Loss: {cur_loss:9} Norm: {norm:9}')
        min_loss = loss
        best_k = k
        best_cluster_index = cluster_index
        best_assignment = assignment

print("----------Cluster Center Info----------")
print(f'Using {best_k} clusters')
print("Cluster Similarity Matrix:")
print(similarity[best_cluster_index, :][:, best_cluster_index])
print(f'Current Loss: {min_loss}')
print(best_assignment.unique(return_counts=True))
print("----------------------------------------")

print("\n\n\n------------Cluster Info-------------")
for i in range(best_k):
    cluster_label = probs[best_cluster_index[i]].argmax(dim=-1).item()
    cluster_context = dataset["label"][best_cluster_index[i]]
    print(f'{i}th cluster idx {best_cluster_index[i]:4} with label (ResNet): '
          f'{model.config.id2label[cluster_label]:10}, context: {cluster_context}')
    member = (best_assignment == i).nonzero(as_tuple=False).squeeze().long()

    if isinstance(member.tolist(), list):
        print(f'Tot: {len(member)}')
        labels = probs[member].argmax(dim=-1).tolist()
        contexts = [dataset["label"][x] for x in member]
        print(f'Label Stats: {Counter(labels)}')
        print(f'Context Stats: {Counter(contexts)}')
        identical_members = list(
            filter(lambda x: x[0] == cluster_label and x[1] == cluster_context, zip(labels, contexts)))
        identical_context_members = list(filter(lambda x: x == cluster_context, contexts))
        print(f'Label & Context Acc: {len(identical_members) / len(member)}')
        print(f'Context Acc:{len(identical_context_members) / len(member)}')
    else:
        print(f'This cluster has center only.')
    print("----------------------------------------")

print('REF: K-Means')
kmeans = KMeans(n_clusters=best_k).fit(features.flatten(start_dim=1, end_dim=len(features.shape) - 1))
for i in range(best_k):
    print(f'{i}th cluster:')
    member = (torch.Tensor(kmeans.labels_) == i).nonzero(as_tuple=False)
    contexts = [dataset["label"][x] for x in member]
    counter = Counter(contexts)
    print(f'Context Acc:{counter.most_common(1)[0][1]/ len(member)}')
    print("----------------------------------------")
