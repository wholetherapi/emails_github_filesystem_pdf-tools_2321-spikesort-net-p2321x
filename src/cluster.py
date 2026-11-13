"""Clustering head for spike assignments."""

import numpy as np


def assign_clusters(embeddings, centroids):
    distances = ((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return distances.argmin(-1)
