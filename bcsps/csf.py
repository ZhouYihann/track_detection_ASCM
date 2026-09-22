# -*- coding: utf-8 -*-
"""
CSF (Cloth Simulation Filter) - a faithful Python port of jianboqi/CSF.

Reference: Zhang et al., "An Easy-to-Use Airborne LiDAR Data Filtering Method
Based on Cloth Simulation", Remote Sensing, 2016.
C++ original: https://github.com/jianboqi/CSF (Apache-2.0)

Internal coordinates (same as the C++):
  - x / grid: the two grid axes; height: the height axis.
  The cloth starts at bbMax(height)+0.05 and falls toward -height under gravity,
  getting pinned (unmovable) where it hits the surface (heightvals). Hence:
    height = -z -> cloth wraps the lower surface from below (classic CSF)
    height = +z -> cloth wraps the upper surface from above (BCSPS's 2nd pass)
"""

import numpy as np
from numba import njit
from scipy import ndimage

# Precomputed displacement tables from Particle.h (used by satisfyConstraintSelf)
SINGLE_MOVE1 = np.array([0., 0.3, 0.51, 0.657, 0.7599, 0.83193, 0.88235,
                         0.91765, 0.94235, 0.95965, 0.97175, 0.98023,
                         0.98616, 0.99031, 0.99322])
DOUBLE_MOVE1 = np.array([0., 0.3, 0.42, 0.468, 0.4872, 0.4949, 0.498,
                         0.4992, 0.4997, 0.4999, 0.4999, 0.5, 0.5, 0.5, 0.5])

# 16 neighbors from the Cloth.cpp constructor (constraint creation order)
# Tuple so numba treats it as a compile-time constant.
_NEIGHBOR_OFFSETS = (
    (-1, 0), (0, -1), (-1, -1), (1, -1), (1, 0), (0, 1), (1, 1), (-1, 1),
    (-2, 0), (0, -2), (-2, -2), (2, -2), (2, 0), (0, 2), (2, 2), (-2, 2),
)

DAMPING = 0.01        # Particle.h
GRAVITY = 0.2         # CSF.cpp
CLOTH_Y_HEIGHT = 0.05 # CSF.cpp: initial cloth height above the bbox max
CLOTH_BUFFER_D = 2    # CSF.cpp: buffer cells around the grid
SMOOTH_THRESHOLD = 0.3        # passed to the Cloth constructor in do_cloth()
HEIGHT_THRESHOLD = 9999.0     # same
MAX_PARTICLE_FOR_POSTPROCESSIN = 50  # Cloth.h


@njit(cache=True)
def _rasterize(pts_x, pts_g, pts_h, origin_x, origin_g, res, W, Hn, cell_mode):
    """Assign each point to a cell and compute each cell's reference height.

    cell_mode: 0 = nearest (C++ semantics: height of the nearest point)
               1 = max (highest point in the cell, for top-surface cloth)
    """
    n = pts_x.shape[0]
    cell_id = np.empty(n, dtype=np.int64)
    planar_dist2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        c = int(round((pts_x[i] - origin_x) / res))
        r = int(round((pts_g[i] - origin_g) / res))
        if c < 0:
            c = 0
        elif c >= W:
            c = W - 1
        if r < 0:
            r = 0
        elif r >= Hn:
            r = Hn - 1
        cell_id[i] = r * W + c
        dx = pts_x[i] - (origin_x + c * res)
        dg = pts_g[i] - (origin_g + r * res)
        planar_dist2[i] = dx * dx + dg * dg

    order = np.argsort(cell_id)
    cell_id_s = cell_id[order]
    heightvals = np.full(W * Hn, -np.inf, dtype=np.float64)

    # member range of each cell
    starts = np.flatnonzero(np.concatenate(
        (np.ones(1, np.bool_), cell_id_s[1:] != cell_id_s[:-1])))
    for s_idx in range(starts.shape[0]):
        s = starts[s_idx]
        e = starts[s_idx + 1] if s_idx + 1 < starts.shape[0] else n
        if cell_mode == 0:  # nearest
            best = s
            best_d = planar_dist2[order[s]]
            for k in range(s + 1, e):
                if planar_dist2[order[k]] < best_d:
                    best_d = planar_dist2[order[k]]
                    best = k
            heightvals[cell_id_s[s]] = pts_h[order[best]]
        else:  # max
            best_h = -np.inf
            for k in range(s, e):
                if pts_h[order[k]] > best_h:
                    best_h = pts_h[order[k]]
            heightvals[cell_id_s[s]] = best_h
    return heightvals


@njit(cache=True)
def _fill_empty_cells(heightvals, W, Hn):
    """Fill empty cells with C++ scanline semantics: rows then columns.

    Leftover empties (edge buffer ring, etc.) propagate inward until filled, so
    particles in empty cells do not free-fall and interpolate absurd heights.
    """
    for r in range(Hn):
        base = r * W
        val = -np.inf
        for c in range(W):
            if heightvals[base + c] > -1e9:
                val = heightvals[base + c]
            elif val > -1e9:
                heightvals[base + c] = val
        val = -np.inf
        for c in range(W - 1, -1, -1):
            if heightvals[base + c] > -1e9:
                val = heightvals[base + c]
            elif val > -1e9:
                heightvals[base + c] = val
    for c in range(W):
        val = -np.inf
        for r in range(Hn):
            if heightvals[r * W + c] > -1e9:
                val = heightvals[r * W + c]
            elif val > -1e9:
                heightvals[r * W + c] = val
        val = -np.inf
        for r in range(Hn - 1, -1, -1):
            if heightvals[r * W + c] > -1e9:
                val = heightvals[r * W + c]
            elif val > -1e9:
                heightvals[r * W + c] = val
    # leftover empties: propagate neighbor values until stable
    for _ in range(20):
        changed = False
        for i in range(W * Hn):
            if heightvals[i] <= -1e9:
                c = i % W
                r = i // W
                v = -np.inf
                if c > 0 and heightvals[i - 1] > v:
                    v = heightvals[i - 1]
                if c < W - 1 and heightvals[i + 1] > v:
                    v = heightvals[i + 1]
                if r > 0 and heightvals[i - W] > v:
                    v = heightvals[i - W]
                if r < Hn - 1 and heightvals[i + W] > v:
                    v = heightvals[i + W]
                if v > -1e9:
                    heightvals[i] = v
                    changed = True
        if not changed:
            break


@njit(cache=True)
def _simulate(pos_h, old_h, movable, heightvals, W, Hn, rigidness,
              iterations, time_step):
    """Cloth simulation main loop: verlet -> constraints -> collision -> convergence."""
    n = pos_h.shape[0]
    dt2 = time_step * time_step
    # CSF.cpp applies gravity*time_step2, then Particle::timeStep multiplies again
    accel = -GRAVITY * dt2 * dt2

    sm = SINGLE_MOVE1[rigidness]
    dm = DOUBLE_MOVE1[rigidness]

    for it in range(iterations):
        # 1) verlet integration
        for i in range(n):
            if movable[i]:
                tmp = pos_h[i]
                pos_h[i] = pos_h[i] + (pos_h[i] - old_h[i]) * (1.0 - DAMPING) + accel
                old_h[i] = tmp
        # 2) constraint relaxation (constraintTimes = rigidness)
        for sweep in range(rigidness):
            for i in range(n):
                ci = i % W
                ri = i // W
                pi = pos_h[i]
                for k in range(16):
                    cj = ci + _NEIGHBOR_OFFSETS[k][0]
                    rj = ri + _NEIGHBOR_OFFSETS[k][1]
                    if cj < 0 or cj >= W or rj < 0 or rj >= Hn:
                        continue
                    j = rj * W + cj
                    corr = pos_h[j] - pi
                    if movable[i] and movable[j]:
                        h = corr * dm
                        pos_h[i] += h
                        pos_h[j] -= h
                    elif movable[i]:
                        pos_h[i] += corr * sm
                    elif movable[j]:
                        pos_h[j] -= corr * sm
        # 3) collision
        for i in range(n):
            if pos_h[i] < heightvals[i]:
                pos_h[i] = heightvals[i]
                movable[i] = False
        # 4) convergence check
        max_diff = 0.0
        for i in range(n):
            if movable[i]:
                d = abs(old_h[i] - pos_h[i])
                if d > max_diff:
                    max_diff = d
        if max_diff != 0.0 and max_diff < 0.005:
            break
    return pos_h, movable


@njit(cache=True)
def _slope_postprocess(pos_h, movable, heightvals, W, Hn):
    """movableFilter: slope post-processing for large movable components.

    Corresponds to C++ findUnmovablePoint + handle_slop_connected.
    """
    n = W * Hn
    comp = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    members = np.empty(n, dtype=np.int64)

    # 4-neighbor directions (matching C++ findUnmovablePoint order: L/R/D/U)
    dc4 = (-1, 1, 0, 0)
    dr4 = (0, 0, -1, 1)

    for i in range(n):
        if (not movable[i]) or comp[i] != -1:
            continue
        # BFS to collect the connected component
        head, tail = 0, 0
        queue[tail] = i
        tail += 1
        comp[i] = 1  # temporary mark (1 for the whole component)
        count = 0
        while head < tail:
            cur = queue[head]
            head += 1
            members[count] = cur
            count += 1
            ci = cur % W
            ri = cur // W
            for k in range(4):
                cj = ci + dc4[k]
                rj = ri + dr4[k]
                if cj < 0 or cj >= W or rj < 0 or rj >= Hn:
                    continue
                j = rj * W + cj
                if movable[j] and comp[j] == -1:
                    comp[j] = 1
                    queue[tail] = j
                    tail += 1
        if count <= MAX_PARTICLE_FOR_POSTPROCESSIN:
            continue

        # findUnmovablePoint: snap members next to a smooth, unmovable neighbor
        edges = np.empty(count, dtype=np.int64)
        nedges = 0
        for m in range(count):
            cur = members[m]
            ci = cur % W
            ri = cur // W
            for k in range(4):
                cj = ci + dc4[k]
                rj = ri + dr4[k]
                if cj < 0 or cj >= W or rj < 0 or rj >= Hn:
                    continue
                j = rj * W + cj
                if not movable[j]:
                    if (abs(heightvals[cur] - heightvals[j]) < SMOOTH_THRESHOLD
                            and pos_h[cur] - heightvals[cur] < HEIGHT_THRESHOLD):
                        pos_h[cur] = heightvals[cur]
                        movable[cur] = False
                        edges[nedges] = cur
                        nedges += 1
                        break
        if nedges == 0:
            continue

        # handle_slop_connected: propagate from the edges inward
        visited = np.full(n, False)
        head, tail = 0, 0
        for e in range(nedges):
            if not visited[edges[e]]:
                visited[edges[e]] = True
                queue[tail] = edges[e]
                tail += 1
        while head < tail:
            cur = queue[head]
            head += 1
            ci = cur % W
            ri = cur // W
            for k in range(4):
                cj = ci + dc4[k]
                rj = ri + dr4[k]
                if cj < 0 or cj >= W or rj < 0 or rj >= Hn:
                    continue
                j = rj * W + cj
                if movable[j] and not visited[j]:
                    if (abs(heightvals[cur] - heightvals[j]) < SMOOTH_THRESHOLD
                            and abs(pos_h[j] - heightvals[j]) < HEIGHT_THRESHOLD):
                        pos_h[j] = heightvals[j]
                        movable[j] = False
                        visited[j] = True
                        queue[tail] = j
                        tail += 1
    return pos_h, movable


def simulate(pts_x, pts_g, pts_h, cloth_resolution=1.0, rigidness=3,
             iterations=500, time_step=0.65, slope_smooth=True,
             cell_mode='nearest', open_window_m=None):
    """Run the cloth simulation. Returns (cloth_heights, origin_x, origin_g, W, Hn, res).

    pts_x / pts_g: grid-plane coords; pts_h: height-axis coords.
    cloth_heights: per-cell cloth height (height axis), shape (Hn, W).

    open_window_m: if not None, morphologically close the rasterized terrain
    (max->min, window in m) before simulating, to erase narrow raised strips
    (e.g. rails, ~7-16 cm wide) so the bottom-up cloth cannot see them.
    """
    pts_x = np.asarray(pts_x, dtype=np.float64).ravel()
    pts_g = np.asarray(pts_g, dtype=np.float64).ravel()
    pts_h = np.asarray(pts_h, dtype=np.float64).ravel()

    res = float(cloth_resolution)
    bb_min_x, bb_max_x = pts_x.min(), pts_x.max()
    bb_min_h, bb_max_h = pts_h.min(), pts_h.max()
    bb_min_g, bb_max_g = pts_g.min(), pts_g.max()

    origin_x = bb_min_x - CLOTH_BUFFER_D * res
    origin_g = bb_min_g - CLOTH_BUFFER_D * res
    origin_h = bb_max_h + CLOTH_Y_HEIGHT

    W = int(np.floor((bb_max_x - bb_min_x) / res)) + 2 * CLOTH_BUFFER_D
    Hn = int(np.floor((bb_max_g - bb_min_g) / res)) + 2 * CLOTH_BUFFER_D
    W = max(W, 1)
    Hn = max(Hn, 1)

    cm = 0 if cell_mode == 'nearest' else 1
    heightvals = _rasterize(pts_x, pts_g, pts_h, origin_x, origin_g,
                            res, W, Hn, cm)
    _fill_empty_cells(heightvals, W, Hn)
    # large holes still empty (e.g. occluded areas): fill with the global median
    still = ~np.isfinite(heightvals)
    if still.any():
        heightvals[still] = np.median(heightvals[~still])

    if open_window_m is not None:
        k = max(3, int(round(open_window_m / res)) | 1)
        h = heightvals.reshape(Hn, W)
        h = np.where(np.isfinite(h), h, -np.inf)
        h = ndimage.maximum_filter(h, size=k)
        h = ndimage.minimum_filter(h, size=k)
        heightvals = h.ravel()

    n = W * Hn
    pos_h = np.full(n, origin_h, dtype=np.float64)
    old_h = pos_h.copy()
    movable = np.ones(n, dtype=np.bool_)

    pos_h, movable = _simulate(pos_h, old_h, movable, heightvals, W, Hn,
                               rigidness, iterations, time_step)

    if slope_smooth:
        pos_h, movable = _slope_postprocess(pos_h, movable, heightvals, W, Hn)

    return pos_h.reshape(Hn, W), origin_x, origin_g, W, Hn, res


@njit(cache=True)
def interp_heights(heights, origin_x, origin_g, W, Hn, res, pts_x, pts_g):
    """Bilinear interpolation (C++ c2cdist) of the cloth height at arbitrary points.

    heights: cloth height grid of shape (Hn, W).
    """
    out = np.empty(pts_x.shape[0], dtype=np.float64)
    max_col = W - 1
    max_row = Hn - 1
    for i in range(pts_x.shape[0]):
        delta_x = pts_x[i] - origin_x
        delta_g = pts_g[i] - origin_g
        col0 = int(delta_x / res)
        row0 = int(delta_g / res)
        if col0 < 0:
            col0 = 0
        elif col0 > max_col:
            col0 = max_col
        if row0 < 0:
            row0 = 0
        elif row0 > max_row:
            row0 = max_row
        col1 = col0 + 1
        row1 = row0 + 1
        if col1 > max_col:
            col1 = max_col
        if row1 > max_row:
            row1 = max_row
        sub_x = (delta_x - col0 * res) / res
        sub_g = (delta_g - row0 * res) / res
        # clamp out-of-grid queries to [0,1]; extrapolation would amplify height steps
        if sub_x < 0.0:
            sub_x = 0.0
        elif sub_x > 1.0:
            sub_x = 1.0
        if sub_g < 0.0:
            sub_g = 0.0
        elif sub_g > 1.0:
            sub_g = 1.0
        fxy = (heights[row0, col0] * (1 - sub_x) * (1 - sub_g)
               + heights[row0, col1] * (1 - sub_x) * sub_g
               + heights[row1, col1] * sub_x * sub_g
               + heights[row1, col0] * sub_x * (1 - sub_g))
        out[i] = fxy
    return out
