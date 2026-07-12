"""Dependency-free k-means with k-means++ initialisation.

Operates on the handful of thousand pixels in a downscaled sprite, so it's fast.
Ported unchanged from the proven reference implementation (reference/pixelize.py) —
any behavioural change here breaks the pixelizer regression test.
"""

from __future__ import annotations

import numpy as np


def kmeans(data: np.ndarray, k: int, iters: int = 40, seed: int = 0):
    """Cluster `data` (n, d) into `k` centers. Returns (centers, labels)."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=data.dtype)
    centers[0] = data[rng.integers(n)]
    d2 = ((data - centers[0]) ** 2).sum(1)
    for i in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers[i] = data[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, ((data - centers[i]) ** 2).sum(1))

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new_labels = dists.argmin(1)
        new_centers = np.array([
            data[new_labels == j].mean(0) if np.any(new_labels == j) else centers[j]
            for j in range(k)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            centers, labels = new_centers, new_labels
            break
        centers, labels = new_centers, new_labels
    return centers, labels
