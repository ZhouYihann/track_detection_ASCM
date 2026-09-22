# -*- coding: utf-8 -*-
"""
BCSPS bidirectional cloth-simulation rail-surface extraction (reproduction).

Paper: Shi, Yang, Kou, Wang, "A fast railway track surface extraction method
based on bidirectional cloth simulated point clouds",
Optics and Lasers in Engineering, 2024. DOI: 10.1016/j.optlaseng.2024.108335
Original program: https://github.com/sz0706/BCSPS (Windows exe; this is a Python port)

Pipeline:
  1. Preprocess  - remove NaN / elevation outliers
  2. Roadbed     - bottom-up cloth (classic CSF) -> roadbed surface zb
  3. Top surface - top-down cloth on points above the roadbed -> zt
  4. Rail        - points well above the roadbed (z - zb > H) and close to zt
"""

import numpy as np

from . import csf


def preprocess(xyz, deep_cut=5.0, high_cut=10.0, max_xy=1e4):
    """Remove NaN, elevation outliers (vs median), and extreme x/y values.

    Tools like VCGLIB can leave ~1e38 dirty coordinates that blow up the cloth
    grid; max_xy clips them (real scan coordinates never exceed +/-1e4 m).

    Returns (float64 points, valid-point indices).
    """
    xyz = np.asarray(xyz)
    ok = np.isfinite(xyz).all(axis=1)
    ok = ok & (np.abs(xyz[:, 0]) <= max_xy) & (np.abs(xyz[:, 1]) <= max_xy)
    if not ok.all():
        z_med = np.median(xyz[ok, 2])
        z_ok = (xyz[:, 2] >= z_med - deep_cut) & (xyz[:, 2] <= z_med + high_cut)
        ok = ok & z_ok
    return xyz[ok].astype(np.float64), ok


def track_surface_extract(xyz, cloth_resolution=0.01, rigidness=3,
                          iterations=500, time_step=0.65,
                          bed_open=0.15, height_diff=0.15,
                          top_tol=0.03, z_cap=1.5, slope_smooth=True,
                          top_cell_mode='max', deep_cut=5.0, high_cut=10.0,
                          verbose=True):
    """Extract rail-surface points via bidirectional cloth simulation.

    Args:
      cloth_resolution: cloth grid resolution (m). ~0.01 for demo scale,
                        ~0.02-0.05 for large scenes.
      rigidness:        1 (flat) / 2 (medium) / 3 (steep), default 3.
      bed_open:         closing window (m) for the roadbed, > rail width.
      height_diff:      H, height above roadbed to count as a raised structure.
      top_tol:          delta, tolerance to the upper cloth surface (m).
      z_cap:            only points within this height above the roadbed feed the
                        top-down simulation (drops catenary, roofs, etc.).
      top_cell_mode:    top-cloth cell height: 'max' (recommended) or 'nearest'.

    Returns:
      mask: bool array marking rail-surface points
      (zb_grid, zt_grid): roadbed and top cloth grids
    """
    n_in = len(xyz)
    xyz, ok = preprocess(xyz, deep_cut, high_cut)
    n = len(xyz)
    if verbose:
        print(f'Input {n_in} points ({n} kept after preprocessing)')

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 1) Roadbed: bottom-up cloth (height axis = -z, i.e. classic CSF).
    # Close the rasterized terrain first to erase narrow raised strips (rails),
    # otherwise the cloth gets pulled up by the rail bottoms.
    if verbose:
        print(f'Roadbed: cloth res {cloth_resolution} m, rigidness {rigidness}, '
              f'closing window {bed_open} m')
    cloth_b, ox, og, W, Hn, res = csf.simulate(
        x, y, -z, cloth_resolution, rigidness, iterations, time_step,
        slope_smooth, cell_mode='nearest', open_window_m=bed_open)
    bed_grid = -cloth_b
    zb = csf.interp_heights(bed_grid, ox, og, W, Hn, res, x, y)
    if verbose:
        print(f'  roadbed z range {zb.min():.3f} ~ {zb.max():.3f} m')

    # 2) Top surface: top-down cloth on points within z_cap above the roadbed.
    if verbose:
        print('Top surface (points within %.1f m above roadbed)' % z_cap)
    top_sel = z <= zb + z_cap
    xt, yt, zt_pts = x[top_sel], y[top_sel], z[top_sel]
    cloth_t, otx, otg, Wt, Hnt, rest = csf.simulate(
        xt, yt, zt_pts, cloth_resolution, rigidness, iterations, time_step,
        slope_smooth, cell_mode=top_cell_mode)
    zt = csf.interp_heights(cloth_t, otx, otg, Wt, Hnt, rest, x, y)
    if verbose:
        print(f'  top surface z range {zt.min():.3f} ~ {zt.max():.3f} m')

    # 3) Rail extraction.
    above_bed = z - zb
    near_top = z - zt
    # Rails sit near the roadbed (within z_cap); higher structures (roofs,
    # catenary) are excluded, and interpolation garbage at cloth holes is avoided.
    m = ((above_bed > height_diff) & (near_top >= -top_tol)
         & (above_bed <= z_cap))
    # Map back to original input indices (same length as input).
    mask = np.zeros(n_in, dtype=bool)
    mask[ok] = m
    if verbose:
        print(f'Rail points {m.sum()} / {n} ({100 * m.mean():.2f}%)')

    return mask, ((bed_grid, ox, og, W, Hn, res),
                  (cloth_t, otx, otg, Wt, Hnt, rest))
