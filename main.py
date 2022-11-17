import json
import sys

import torch
from collections import Counter
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from datasets import load_dataset, Image, Dataset
from matplotlib import pyplot as plt
from torchmetrics.functional import pairwise_cosine_similarity
from torch.nn.functional import softmax


def flattened_cosine_similarity(x, y=None):
    flattened_x = x.flatten(start_dim=1, end_dim=len(x.shape) - 1)
    if y is None:
        return pairwise_cosine_similarity(flattened_x, zero_diagonal=False)
    else:
        flattened_y = y.flatten(start_dim=1, end_dim=len(y.shape) - 1)
        return pairwise_cosine_similarity(flattened_x, flattened_y, zero_diagonal=False)


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
cluster_candidate = probs_max.indices[probs_max.values > 0.9999]
poor_image_index = probs_max.indices[probs_max.values < 0.90].tolist()

with open(f'./output/poor_image.txt', 'w') as f:
    for idx in poor_image_index:
        print(dataset["train"]["name"][idx], file=f)
print("Dump poor images")

similarity = flattened_cosine_similarity(features, features)
distance = (1 - similarity[cluster_candidate, :][:, cluster_candidate]).sum(dim=0)

min_loss = 1e30
best_k = 3
best_cluster_index = None
best_assignment = None
for k in range(2, len(distance)):
    cluster_index = cluster_candidate[distance.topk(k).indices]
    filtered_similarity = similarity[:, cluster_index]
    assignment = filtered_similarity.argmax(dim=1)
    cur_loss = 0
    for i in range(k):
        group_i_index = (assignment == i).nonzero(as_tuple=False)
        item_loss = 1 - filtered_similarity[group_i_index, i]
        cur_loss += item_loss.mean().item()

    cur_loss = cur_loss / k
    cluster_similarity = similarity[cluster_index, :][:, cluster_index]
    norm = torch.linalg.norm(cluster_similarity)
    cur_loss = cur_loss + norm / 10
    if cur_loss < min_loss:
        print(f'#Cluster: {k:3} Loss: {cur_loss:9} Norm: {norm:9}')
        min_loss = cur_loss
        best_k = k
        best_cluster_index = cluster_index
        best_assignment = assignment

print("----------Cluster Info----------")
print(f'Using {best_k} clusters:')
print(dataset["train"][best_cluster_index]["name"])
print("Cluster Similarity Matrix:")
print(similarity[best_cluster_index, :][:, best_cluster_index])
print(f'Current Loss: {min_loss}')
print(best_assignment.unique(return_counts=True))
print("--------------------------------")

s = []
with open(f'./output/', 'w') as f, open(f'output/cluster.txt', 'w') as g:
    for name in dataset["train"][best_cluster_index]["name"]:
        print(name, file=g)

    for (i, (name, c)) in enumerate(zip(dataset["train"]["name"], best_assignment)):
        s.append(similarity[i, best_cluster_index[c]])
        # print(f'{i:3} & {best_cluster_index[c]:3}: {similarity[i, best_cluster_index[c]]: 9}')
        print(name + f' {c}', file=f)
print(torch.Tensor(s).sort().values)
