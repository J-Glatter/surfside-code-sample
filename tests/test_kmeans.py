from __future__ import annotations

import numpy as np

from spriteforge.kmeans import kmeans


def _three_blobs(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return np.concatenate([c + rng.normal(0, 0.3, size=(50, 2)) for c in centers])


def test_recovers_separated_clusters():
    data = _three_blobs()
    centers, labels = kmeans(data, k=3, seed=0)
    # each true blob maps to exactly one recovered cluster
    for start in (0, 50, 100):
        blob_labels = labels[start:start + 50]
        assert len(set(blob_labels.tolist())) == 1
    # recovered centers close to the true ones (order-independent)
    true = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    for t in true:
        assert np.min(np.linalg.norm(centers - t, axis=1)) < 0.5


def test_deterministic_for_seed():
    data = _three_blobs()
    c1, l1 = kmeans(data, k=3, seed=7)
    c2, l2 = kmeans(data, k=3, seed=7)
    assert np.array_equal(c1, c2)
    assert np.array_equal(l1, l2)


def test_k_equals_one():
    data = _three_blobs()
    centers, labels = kmeans(data, k=1, seed=0)
    assert centers.shape == (1, 2)
    assert np.allclose(centers[0], data.mean(0))
    assert np.all(labels == 0)
