# -*- coding: utf-8 -*-
"""PCA linearity post-processing: keep locally linear points (rails), drop
planar false positives (platforms, etc.).

Corresponds to the paper's "PCA spatial dimension features (k = 20, threshold 0.5)".
"""

import numpy as np
from scipy.spatial import cKDTree


def linearity_mask(points, k=20, chunk=200000, threshold=0.5):
    """PCA on each point's k-neighbors; returns (linearity, mask).

    linearity = (l1 - l2) / l1 for neighborhood covariance eigenvalues l1 >= l2 >= l3.
    Linear structures (rails) have high linearity; planar ones (platforms) low.
    """
    tree = cKDTree(points)
    lin = np.empty(len(points), dtype=np.float64)
    for s in range(0, len(points), chunk):
        e = min(s + chunk, len(points))
        _, idx = tree.query(points[s:e], k=k, workers=-1)
        # idx: (m, k); column 0 is the point itself
        nb = points[idx]              # (m, k, 3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        # 3x3 covariance = J^T J, computed batchwise with einsum
        cov = np.einsum('mki,mkj->mij', nb, nb) / k
        w = np.linalg.eigvalsh(cov)   # ascending l3, l2, l1
        lin[s:e] = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2], 1e-12)
    return lin, lin >= threshold
