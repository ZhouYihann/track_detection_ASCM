# -*- coding: utf-8 -*-
"""Rail-top ridge post-processing: keep only the narrow rail bands.

The bidirectional criterion (z - zb > H and z near the upper surface) selects
everything above the roadbed - rails, sleeper tops, slab-track beds, guard rails.
When those surfaces sit less than H below the rail top (slab track is usually
only 0.10-0.15 m lower), they get selected as whole patches.

Key: the rails are the highest narrow strips. Per short along-track slice we
compute the cross-section "highest surface" profile z_top(t), and keep only the
narrow band within h_tol of the local rail-top reference height. The middle band
lies >0.10 m below the rail top and is dropped.
"""

import numpy as np


def rail_top_filter(points, h_tol=0.08, w_max=0.45,
                    along_bin=0.25, cross_bin=0.02, ref_window=2.0):
    """Keep only the narrow rail bands.

    points: (N,3) already-extracted raised-structure points.
    h_tol: max height difference (m) from the rail-top reference to keep.
    w_max: max band width (m); wider high bands (roadbed, etc.) are dropped.
    along_bin / cross_bin: along / lateral bin sizes (m).
    ref_window: along-track smoothing window (m) for the rail-top reference.

    Returns a bool mask.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)

    # along direction: principal direction of the xy plane
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    cov = xy.T @ xy
    w, v = np.linalg.eigh(cov)
    along = v[:, 1]                     # principal eigenvector
    if along[0] < 0:
        along = -along
    cross = np.array([-along[1], along[0]])
    s = xy @ along
    t = xy @ cross

    s -= s.min()
    t -= t.min()
    ns = int(s.max() / along_bin) + 1
    nt = int(t.max() / cross_bin) + 1
    si = np.minimum((s / along_bin).astype(np.int64), ns - 1)
    ti = np.minimum((t / cross_bin).astype(np.int64), nt - 1)
    cell = si * nt + ti

    # highest z per (along, lateral) cell
    prof = np.full(ns * nt, -np.inf)
    np.maximum.at(prof, cell, pts[:, 2])
    prof = prof.reshape(ns, nt)
    prof[~np.isfinite(prof)] = np.nan

    # rail-top reference: per-slice highest surface, then a rolling median
    # along the track (tolerates slices where the rail is missing)
    row_max = np.nanmax(prof, axis=1)
    half = max(1, int(ref_window / along_bin))
    ref = np.empty(ns)
    for i in range(ns):
        lo, hi = max(0, i - half), min(ns, i + half + 1)
        seg = row_max[lo:hi]
        ref[i] = np.nanmedian(seg) if np.any(np.isfinite(seg)) else np.nan

    # per slice: keep continuous narrow bands close to the reference height
    keep_cell = np.zeros((ns, nt), dtype=bool)
    for i in range(ns):
        if not np.isfinite(ref[i]):
            continue
        band = np.isfinite(prof[i]) & (prof[i] >= ref[i] - h_tol)
        # contiguous runs
        d = np.diff(np.concatenate(([0], band.view(np.int8), [0])))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0]
        for a, b in zip(starts, ends):
            if (b - a) * cross_bin <= w_max:
                keep_cell[i, a:b] = True

    return keep_cell[si, ti]


def apply_filter(points, mask):
    return np.asarray(points)[mask]
