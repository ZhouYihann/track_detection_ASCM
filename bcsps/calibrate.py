# -*- coding: utf-8 -*-
"""Per-segment calibration: find "up" -> find rails -> measure gauge (scale) -> build canonical frame.

Each COLMAP-reconstructed segment has an arbitrary, unoriented, arbitrary-scale
coordinate system. The track itself provides a natural scale: gauge = 1435 mm
(standard), and the two rails are the most prominent continuous strips.

Steps:
  1. Local-normal mode -> "up" direction (ground dominates in corridors).
  2. Upright -> ground plane (grid low-quantile) -> h = z - ground.
  3. Histogram of h -> rail-top height.
  4. Projection histogram (Hough) over candidate angles -> track direction + rail positions.
  5. gauge (units) / 1.435 m = scale; canonical frame: origin = track start
     center, x = along, y = lateral, z = up, in meters.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import find_peaks

GAUGE_M = 1.435          # standard gauge (m)
UP_QUANTILE = 10         # low quantile for ground estimation
GRID = 0.1               # ground grid (input units)


# ---------------------------------------------------------------- basics

def estimate_up(P, nq=80000, k=25, seed=0):
    """Spherical mode of local normals -> plane normal (up candidate, sign undetermined)."""
    P = np.asarray(P, dtype=np.float64)
    tree = cKDTree(P)
    rng = np.random.RandomState(seed)
    q = P[rng.choice(len(P), min(nq, len(P)), replace=False)]
    _, idx = tree.query(q, k=k, workers=-1)
    nb = P[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('mki,mkj->mij', nb, nb)
    _, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0].copy()                 # smallest eigenvector = local normal
    nrm[nrm[:, 2] < 0] *= -1                # flip to upper hemisphere
    az = np.arctan2(nrm[:, 1], nrm[:, 0])
    el = np.arcsin(np.clip(nrm[:, 2], -1, 1))
    H, xe, ye = np.histogram2d(az, el, bins=[72, 36],
                               range=[[-np.pi, np.pi], [0, np.pi / 2]])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    a = 0.5 * (xe[i] + xe[i + 1])
    e = 0.5 * (ye[j] + ye[j + 1])
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def basis_from_up(up):
    """Build an orthonormal basis (e1, e2, up) from the up direction."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    tmp = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2, up


def refine_up(P, up0, iterations=3, grid=GRID, min_pts=30, verbose=False):
    """Refine the up direction by repeatedly fitting the ground plane.

    The normal-mode estimate is only accurate to a few degrees, which over tens of
    meters causes decimetre-scale drift that swamps the rails (~0.2 m above the
    roadbed). The ground is the largest plane, so its fitted normal is used.
    """
    up = np.asarray(up0, dtype=np.float64)
    up /= np.linalg.norm(up)
    for it in range(iterations):
        e1, e2, up = basis_from_up(up)
        A, B, C = P @ e1, P @ e2, P @ up
        gx = np.floor(A / grid).astype(np.int64)
        gy = np.floor(B / grid).astype(np.int64)
        gid = (gx - gx.min()) * 1000000 + (gy - gy.min())
        ug, inv = np.unique(gid, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(ug))
        good = cnt >= min_pts
        if good.sum() < 20:
            break
        cA = np.array([A[inv == k].mean() for k in np.nonzero(good)[0]])
        cB = np.array([B[inv == k].mean() for k in np.nonzero(good)[0]])
        cC = np.array([np.percentile(C[inv == k], UP_QUANTILE)
                       for k in np.nonzero(good)[0]])
        # robust plane fit (iteratively drop high-residual cells)
        keep = np.ones(len(cA), dtype=bool)
        coef = None
        for _ in range(4):
            if keep.sum() < 10:
                break
            M = np.column_stack([cA[keep], cB[keep], np.ones(keep.sum())])
            coef, *_ = np.linalg.lstsq(M, cC[keep], rcond=None)
            resid = np.abs(cC - (coef[0] * cA + coef[1] * cB + coef[2]))
            thr = max(np.percentile(resid[keep], 70) * 1.5, 1e-4)
            keep = resid < thr
        if coef is None:
            break
        # plane normal (in the current frame)
        n = np.array([-coef[0], -coef[1], 1.0])
        n /= np.linalg.norm(n)
        # back to world frame: up component = up, e1/e2 components = n[0], n[1]
        up = n[0] * e1 + n[1] * e2 + n[2] * up
        up /= np.linalg.norm(up)
        if verbose:
            print(f'    refine {it+1}: residual tilt {np.degrees(np.arctan(np.hypot(coef[0], coef[1]))):.3f} deg')
    return up


def ground_surface(A, B, C, grid=GRID, quantile=UP_QUANTILE):
    """Grid low-quantile surface -> per-point height above ground h."""
    gx = np.floor(A / grid).astype(np.int64)
    gy = np.floor(B / grid).astype(np.int64)
    gid = (gx - gx.min()) * 1000000 + (gy - gy.min())
    ug, inv = np.unique(gid, return_inverse=True)
    gq = np.array([np.percentile(C[inv == k], quantile) for k in range(len(ug))])
    # median smoothing over neighbor cells (fills holes)
    return C - gq[inv], gq, ug, inv


def _hill(x, lo, hi, soft=0.06):
    """1 inside [lo,hi], decaying to 0 over a soft width outside."""
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(max(0.0, 1.0 - d / soft))


def _peak_width(hist, k, binw):
    """Full width at half maximum of a peak (in units)."""
    half = 0.5 * hist[k]
    l = k
    while l > 0 and hist[l] > half:
        l -= 1
    r = k
    while r < len(hist) - 1 and hist[r] > half:
        r += 1
    return (r - l) * binw


def rail_pair_score(proj, h, edges, peaks, hist, min_gap_frac=0.08):
    """Score which pair of peaks best matches two rails; return the best pair.

    Criteria (all scale-invariant):
      - both lines rise p above the ground between them, with p/gap ~ 0.10-0.30
        (real: rail top 0.2-0.35 m above roadbed, gauge 1.435 m -> 0.14-0.24)
      - the two lines have similar heights (rails of one pair are level)
      - peaks are narrow (rail width ~0.07 m, the narrowest strip in projection)
    """
    if len(peaks) < 2:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = edges[1] - edges[0]
    span = edges[-1] - edges[0]
    best = None
    idx = list(peaks)
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            g = centers[j] - centers[i]
            if g < min_gap_frac * span:
                continue
            m1 = np.abs(proj - centers[i]) < 0.12 * g
            m2 = np.abs(proj - centers[j]) < 0.12 * g
            mb = np.abs(proj - 0.5 * (centers[i] + centers[j])) < 0.2 * g
            if m1.sum() < 30 or m2.sum() < 30 or mb.sum() < 30:
                continue
            h1 = float(np.percentile(h[m1], 90))
            h2 = float(np.percentile(h[m2], 90))
            hb = float(np.percentile(h[mb], 90))
            p = min(h1, h2) - hb
            if p <= 0:
                continue
            ratio = p / g
            sym = max(0.0, 1.0 - abs(h1 - h2) / max(h1, h2, 1e-9))
            # peak width: rails should be much narrower than the gauge
            w1 = _peak_width(hist, i, binw)
            w2 = _peak_width(hist, j, binw)
            narrow = _hill(max(w1, w2) / g, 0.0, 0.22)
            score = (_hill(ratio, 0.10, 0.30) * sym ** 2 * narrow
                     * np.sqrt(max(p, 0)))
            if score <= 0:
                continue
            if best is None or score > best['score']:
                best = dict(score=float(score), i=i, j=j, g=float(g),
                            s1=float(centers[i]), s2=float(centers[j]),
                            h1=h1, h2=h2, hb=hb, p=float(p), ratio=float(ratio),
                            w1=float(w1), w2=float(w2))
    return best


def track_direction_hough(A, B, h, angles=None, nbins=500):
    """Scan candidate angles for the one that best matches two rails.

    Returns (score, theta, projection, histogram edges, peak positions, best pair).
    """
    a = np.asarray(A)
    b = np.asarray(B)
    if angles is None:
        angles = np.arange(0, 180, 1.0)
    best = None
    lo, hi = np.percentile(a, 0.2), np.percentile(a, 99.8)
    lo2, hi2 = np.percentile(b, 0.2), np.percentile(b, 99.8)
    rng = float(np.hypot(hi - lo, hi2 - lo2))
    for th in angles:
        r = np.deg2rad(th)
        nvec = np.array([-np.sin(r), np.cos(r)])   # projection (lateral) direction
        proj = a * nvec[0] + b * nvec[1]
        p1, p99 = np.percentile(proj, 0.5), np.percentile(proj, 99.5)
        edges = np.linspace(p1, p99, nbins + 1)
        hist, _ = np.histogram(proj, bins=edges)
        hs = np.convolve(hist, np.ones(3) / 3, mode='same')
        pk, _ = find_peaks(hs, prominence=hs.max() * 0.06)
        if len(pk) < 2:
            continue
        # keep only the top 8 peaks for pairing
        pk = pk[np.argsort(hs[pk])[::-1][:8]]
        pair = rail_pair_score(proj, h, edges, np.sort(pk), hs)
        if pair is None:
            continue
        # also score the sharpness of the peaks themselves
        sharp = 0.5 * (hs[pair['i']] + hs[pair['j']]) / max(hs.max(), 1e-9)
        sc = pair['score'] * (0.5 + 0.5 * sharp)
        if best is None or sc > best[0]:
            best = (float(sc), float(th), proj, edges, np.sort(pk), pair)
    return best


def _refine_direction(A, B, h, s1, s2, g, n_iter=4):
    """Refine track direction and rail positions via orthogonal regression on rail points.

    Only points near the two rails are used (clean straight structures), avoiding
    drift from other structures. Returns (along unit vector d, lateral n, (S1, S2)).
    """
    r = 0.0
    d = np.array([1.0, 0.0])
    n = np.array([0.0, 1.0])
    S1, S2 = s1, s2
    for _ in range(n_iter):
        S = A * n[0] + B * n[1]
        L = A * d[0] + B * d[1]
        m1 = np.abs(S - S1) < 0.15 * g
        m2 = np.abs(S - S2) < 0.15 * g
        if m1.sum() < 100 or m2.sum() < 100:
            break
        # orthogonal regression (dS/dL) per rail group; average as direction update
        slopes = []
        newS = []
        for m in (m1, m2):
            Lm, Sm = L[m], S[m]
            Lc, Sc = Lm - Lm.mean(), Sm - Sm.mean()
            den = (Lc ** 2).sum()
            if den <= 0:
                continue
            slopes.append((Lc * Sc).sum() / den)
            newS.append(float(np.median(Sm)))
        if len(slopes) < 2:
            break
        ang = float(np.arctan(np.mean(slopes)))    # direction correction angle
        r += ang
        d = np.array([np.cos(r), np.sin(r)])
        n = np.array([-d[1], d[0]])
        # recompute lateral rail positions with the new direction
        S1, S2 = (float(np.percentile(A[m1] * n[0] + B[m1] * n[1], 50)),
                  float(np.percentile(A[m2] * n[0] + B[m2] * n[1], 50)))
        if S1 > S2:
            S1, S2 = S2, S1
    return d, n, (S1, S2)


def _subbin_center(hs, centers, k, w=3):
    """Sub-bin refinement of a peak position: intensity-weighted centroid over +/-w bins."""
    lo = max(0, k - w)
    hi = min(len(hs), k + w + 1)
    seg = hs[lo:hi].astype(np.float64)
    if seg.sum() <= 0:
        return float(centers[k])
    return float((seg * centers[lo:hi]).sum() / seg.sum())


def detect_gauge_by_slabs(A, B, slab=1.0, binw=0.02, top_k=6,
                          min_frac=0.02, max_frac=0.6):
    """Find rails per cross-section via point density, gauge from its constancy along the track.

    The rails are the two densest, equally-spaced parallel narrow lines; other
    structures may form density peaks but their separations vary along the track.
      1. per-slab density peaks (top_k)
      2. collect separations of all peak pairs (weight = product of peak heights)
      3. the histogram mode is the gauge

    Returns (gauge_units, sigma, n_votes, info) or None.
    """
    A = np.asarray(A); B = np.asarray(B)
    lo, hi = np.percentile(A, 5), np.percentile(A, 95)
    edges = np.arange(np.percentile(B, 0.5), np.percentile(B, 99.5), binw)
    if len(edges) < 20:
        return None
    span = edges[-1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    votes = []
    details = []
    for a0 in np.arange(lo, hi - slab, slab * 0.5):
        m = (A >= a0) & (A < a0 + slab)
        if m.sum() < 500:
            continue
        hist, _ = np.histogram(B[m], bins=edges)
        hs = np.convolve(hist, np.ones(3) / 3, mode='same')
        if hs.max() <= 0:
            continue
        pk, _ = find_peaks(hs, prominence=hs.max() * 0.15)
        if len(pk) < 2:
            continue
        # rails are the two densest narrow lines -> take the top two peaks
        top2 = pk[np.argsort(hs[pk])[::-1][:2]]
        top2 = np.sort(top2)
        c1 = _subbin_center(hs, centers, top2[0])
        c2 = _subbin_center(hs, centers, top2[1])
        g = c2 - c1
        if g < max(min_frac * span, 3 * binw) or g > max_frac * span:
            continue
        votes.append(g)
        details.append((a0, c1, c2, g))
    if len(votes) < 5:
        return None
    votes = np.array(votes)
    # median + MAD to drop outlier slabs (occluded or contaminated sections)
    med = float(np.median(votes))
    mad = float(np.median(np.abs(votes - med))) + 1e-9
    keep = np.abs(votes - med) < 3.5 * 1.4826 * mad
    if keep.sum() < 3:
        keep = np.ones(len(votes), dtype=bool)
    gauge = float(np.median(votes[keep]))
    sigma = float(np.std(votes[keep]))
    return gauge, sigma, int(keep.sum()), dict(votes=votes, details=details,
                                               keep=keep)


def track_rails(A, B, gauge, slab=1.0, binw=0.02, max_miss=8, min_pts=80):
    """Track the two rail centerlines slab by slab (tolerating gaps via extrapolation).

    Returns (a_c, s_left, s_right): along-track coordinate and lateral rail positions per slab.
    """
    A = np.asarray(A); B = np.asarray(B)
    lo, hi = np.percentile(A, 2), np.percentile(A, 98)
    edges = np.arange(np.percentile(B, 0.5), np.percentile(B, 99.5), binw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = []
    prev = None
    d_prev = (0.0, 0.0)          # per-slab rail displacement from the last found slab
    miss = 0
    for a0 in np.arange(lo, hi - slab, slab * 0.5):
        m = (A >= a0) & (A < a0 + slab)
        best = None
        if m.sum() >= min_pts:
            hist, _ = np.histogram(B[m], bins=edges)
            hs = np.convolve(hist, np.ones(3) / 3, mode='same')
            pk, _ = find_peaks(hs, prominence=hs.max() * 0.10)
            for ii in range(len(pk)):
                for jj in range(ii + 1, len(pk)):
                    g = centers[pk[jj]] - centers[pk[ii]]
                    if abs(g - gauge) > 0.35 * gauge:
                        continue
                    sc = abs(g - gauge)
                    if prev is not None:
                        sc += 0.5 * (abs(centers[pk[ii]] - prev[0]) +
                                     abs(centers[pk[jj]] - prev[1]))
                    if best is None or sc < best[0]:
                        best = (sc, centers[pk[ii]], centers[pk[jj]])
        if best is None:
            # no pair found: continue with extrapolated positions (max max_miss consecutive)
            if prev is None or miss >= max_miss:
                if prev is not None:
                    break
                continue
            miss += 1
            p1 = prev[0] + d_prev[0]
            p2 = prev[1] + d_prev[1]
            out.append((a0 + slab / 2, p1, p2))
            prev = (p1, p2)
            continue
        if prev is not None:
            d_prev = (best[1] - prev[0], best[2] - prev[1])
        miss = 0
        prev = (best[1], best[2])
        out.append((a0 + slab / 2, best[1], best[2]))
    if not out:
        return None
    arr = np.array(out)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def refine_rails(A, B, h, d, n, s1, s2, n_iter=4):
    """Sub-bin refinement of rail positions and direction.

    Coarse histogram peaks are limited by bin quantization, giving up to ~20% gauge
    error. Use centroid + line fit for continuous positions:
      - rail points = those with h in [0.10g, 0.35g] (scale-invariant ratio band)
      - each iteration: new positions = medians of band points; update direction
        via orthogonal regression on both groups.
    """
    S1, S2 = float(min(s1, s2)), float(max(s1, s2))
    for _ in range(n_iter):
        g = S2 - S1
        if g <= 0:
            break
        S = A * n[0] + B * n[1]
        L = A * d[0] + B * d[1]
        band = (h > 0.08 * g) & (h < 0.40 * g)
        m1 = band & (np.abs(S - S1) < 0.30 * g)
        m2 = band & (np.abs(S - S2) < 0.30 * g)
        if m1.sum() < 100 or m2.sum() < 100:
            break
        # new positions: median of in-band projections (robust centroid)
        newS1 = float(np.median(S[m1]))
        newS2 = float(np.median(S[m2]))
        # direction: orthogonal regression dS/dL per group
        slopes = []
        for m in (m1, m2):
            Lm, Sm = L[m], S[m]
            Lc, Sc = Lm - Lm.mean(), Sm - Sm.mean()
            den = (Lc ** 2).sum()
            if den > 0:
                slopes.append((Lc * Sc).sum() / den)
        if slopes:
            ang = float(np.arctan(np.mean(slopes)))
            ct, st = np.cos(ang), np.sin(ang)
            R2 = np.array([[ct, st], [-st, ct]])
            d = R2 @ d
            n = np.array([-d[1], d[0]])
            # recompute positions after direction update
            S = A * n[0] + B * n[1]
            newS1 = float(np.median(S[m1]))
            newS2 = float(np.median(S[m2]))
        if newS2 - newS1 <= 0:
            break
        S1, S2 = newS1, newS2
    return d, n, (S1, S2)


# ---------------------------------------------------------------- calibration

class SegmentCalibration:
    """Calibration result for one point-cloud segment."""

    def __init__(self, name, R, scale, origin, up, along, cross,
                 gauge_units, rail_h, length_m, score, diag):
        self.name = name
        self.R = R              # 3x3 rotation (input coords -> canonical frame)
        self.scale = scale      # input units -> meters
        self.origin = origin    # canonical-frame origin in input coords
        self.up = up
        self.along = along
        self.cross = cross
        self.gauge_units = gauge_units
        self.rail_h = rail_h
        self.length_m = length_m
        self.score = score
        self.diag = diag        # diagnostics (sampled cloud, rail positions, ...)

    def transform(self, P):
        """Transform input points P into the canonical frame (meters)."""
        return (np.asarray(P, dtype=np.float64) - self.origin) @ self.R.T * self.scale


def detect_trim_point(cal, slope=0.15, min_keep=0.5):
    """Detect where the rail tracking jumps abruptly at the end (mistracking).

    Criterion: lateral slope |dy/dx| > slope between adjacent slabs (normal rail
    curvature is far gentler). Do not use "deviation from a straight trend" since
    curved tracks deviate by design. Returns the trim point (canonical x, m), or None.
    """
    a_c, s_l, s_r = cal.a_c, cal.s_l, cal.s_r
    a0 = float(getattr(cal, 'a_origin', a_c.min()))
    x_m = (a_c - a0) * cal.scale
    cen = 0.5 * (s_l + s_r)
    y = (cen - cen[0]) * cal.scale
    if len(x_m) < 8:
        return None
    dy = np.diff(y)
    dx = np.diff(x_m)
    dx[dx <= 0] = 1e-9
    bad = np.nonzero(np.abs(dy / dx) > slope)[0]
    if len(bad) == 0:
        return None
    idx = int(bad[0])              # last good slab before the jump
    if idx < 3:
        return None
    x_trim = float(x_m[idx])
    if x_trim < min_keep:
        return None
    return x_trim


def up_from_rails(P, M, d, nn, a_c, s_l, s_r, gauge, up_cur, slab=0.5):
    """Determine up from the rails themselves (normal of the track plane).

    Two parallel rails define the track plane; its normal is the true up, far more
    reliable than a ground fit (a cross-slope can skew up by tens of degrees).
    Returns (up_new, angle change) or (None, 0).
    """
    L = M @ d
    S = M @ nn
    p_l, p_r = [], []
    for a0, sl, sr in zip(a_c, s_l, s_r):
        m = np.abs(L - a0) < slab
        if m.sum() < 30:
            continue
        m1 = m & (np.abs(S - sl) < 0.25 * gauge)
        m2 = m & (np.abs(S - sr) < 0.25 * gauge)
        if m1.sum() > 20 and m2.sum() > 20:
            p_l.append(P[m1].mean(axis=0))
            p_r.append(P[m2].mean(axis=0))
    if len(p_l) < 5:
        return None, 0.0
    p_l = np.array(p_l)
    p_r = np.array(p_r)
    both = np.vstack([p_l, p_r])
    c = both - both.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    dir3 = vt[0]                                  # track longitudinal (3D)
    sep = (p_r - p_l).mean(axis=0)                # line between the two rails
    sep -= dir3 * (sep @ dir3)
    up_new = np.cross(dir3, sep)
    nrm = np.linalg.norm(up_new)
    if nrm < 1e-9:
        return None, 0.0
    up_new /= nrm
    # sign: more points should lie below the track plane (roadbed/ground)
    cen_pts = P[np.abs(L - np.median(L)) < 3.0]
    if len(cen_pts) > 100:
        side = (cen_pts - both.mean(axis=0)) @ up_new
        n_above = (side > 0).sum()
        n_below = (side < 0).sum()
        if n_above > n_below:
            up_new = -up_new
    if up_new @ up_cur < 0:
        up_new = -up_new
    ang = float(np.degrees(np.arccos(np.clip(abs(up_new @ up_cur), -1, 1))))
    return up_new, ang


def calibrate(P, name='', verbose=True):
    """Calibrate one point-cloud segment (sampled Nx3).

    Try several up candidates (normal mode / ground refinement / PCA) with both
    signs; score by number of slabs with detected rails and gauge consistency.
    Track longitudinal = principal direction in the horizontal plane.
    """
    P = np.asarray(P, dtype=np.float64)
    up_norm = estimate_up(P)
    up_ref = refine_up(P, up_norm)
    # third candidate: the thinnest axis (a corridor is a thin slab, its normal is up).
    # large wall surfaces can bias the normal mode (observed up to 88 deg off).
    c0 = P - P.mean(axis=0)
    w0, v0 = np.linalg.eigh(c0.T @ c0 / len(c0))
    up_pca = v0[:, 0]
    up_cands = [up_norm, up_ref, up_pca]

    def _try(up0, tag, sign):
        up = up0 * sign
        e1, e2, up = basis_from_up(up)
        M = np.column_stack([P @ e1, P @ e2])
        M = M - M.mean(axis=0)
        w2, v2 = np.linalg.eigh(M.T @ M)
        d = v2[:, 1]                     # along-track (principal direction)
        nn = np.array([-d[1], d[0]])     # lateral
        res = detect_gauge_by_slabs(M @ d, M @ nn)
        if res is None:
            return None
        gauge, sigma, nvotes, info = res
        score = nvotes / (1.0 + sigma / max(gauge, 1e-9))
        # up-down check: rails must be above the ground between them (top up).
        # penalize heavily if flipped, else the track comes out upside down.
        C = P @ up
        S_ = M @ nn
        up_ok = True
        if info.get('details'):
            s1_ = float(np.median([t[1] for t in info['details']]))
            s2_ = float(np.median([t[2] for t in info['details']]))
            g_ = max(s2_ - s1_, 1e-9)
            on_rail = (np.abs(S_ - s1_) < 0.30 * g_) | \
                      (np.abs(S_ - s2_) < 0.30 * g_)
            between = np.abs(S_ - 0.5 * (s1_ + s2_)) < 0.25 * g_
            if on_rail.sum() > 100 and between.sum() > 100:
                up_ok = (np.median(C[on_rail]) > np.median(C[between]))
        if not up_ok:
            score *= 0.2
        # strong flip check: rails should be in the upper half of the cross-section.
        # rail-vs-between alone does not stop flips (after a flip, another pair
        # of structures satisfies it too); a "+z up" prior also fails because
        # some input files are z-down (part9). This check is purely geometric.
        if info.get('details'):
            mid_all = float(np.median(C))
            rail_hi = float(np.median(C[on_rail])) if on_rail.sum() > 100 else mid_all
            if rail_hi < mid_all - 0.01:
                score *= 0.15
        return dict(tag=tag, sign=sign, up=up, e1=e1, e2=e2, d=d, nn=nn,
                    gauge=gauge, sigma=sigma, nvotes=nvotes, score=score,
                    info=info, M=M, up_ok=up_ok)

    best = None
    for it in range(3):
        best_it = None
        for up0 in up_cands:
            for sign in (1, -1):
                cand = _try(up0, 'it%d' % it, sign)
                if cand is not None and (best_it is None or
                                         cand['score'] > best_it['score']):
                    best_it = cand
        if best_it is None:
            break
        tr = track_rails(best_it['M'] @ best_it['d'],
                         best_it['M'] @ best_it['nn'], best_it['gauge'])
        if tr is not None:
            best_it['a_c'], best_it['s_l'], best_it['s_r'] = tr
            if best is None or best_it['score'] > best['score']:
                best = best_it
            # refine up from the rail plane (a ground fit can be skewed by cross-slope)
            if it < 2:
                up_new, ang = up_from_rails(
                    P, best_it['M'], best_it['d'], best_it['nn'],
                    tr[0], tr[1], tr[2], best_it['gauge'], best_it['up'])
                if up_new is not None and ang > 0.5:
                    if verbose:
                        print(f'    [{name}] refined up from track plane: {ang:.2f} deg')
                    up_cands = [up_new]
                    continue
        elif best is None:
            best = best_it
        break

    if best is None or 'a_c' not in best:
        raise RuntimeError(f'{name}: calibration failed (no rails detected)')

    gauge_units = best['gauge']
    scale = GAUGE_M / gauge_units

    tr = (best['a_c'], best['s_l'], best['s_r'])
    if tr is None:
        raise RuntimeError(f'{name}: rail tracking failed')
    a_c, s_l, s_r = tr

    # canonical basis: x = along, y = lateral, z = up
    up = best['up']
    along = best['d'][0] * best['e1'] + best['d'][1] * best['e2']
    cross = np.cross(up, along)
    cross /= np.linalg.norm(cross)
    along = np.cross(cross, up)
    R = np.vstack([along, cross, up])

    # origin: track center at the segment start (min along-track)
    a0 = float(a_c.min())
    i0 = int(np.argmin(a_c))
    s_cen0 = 0.5 * (s_l[i0] + s_r[i0])
    origin_ab = (a0 * best['d'][0] + s_cen0 * best['nn'][0],
                 a0 * best['d'][1] + s_cen0 * best['nn'][1])
    origin = origin_ab[0] * best['e1'] + origin_ab[1] * best['e2']
    up_vals = P @ up
    origin = origin + np.percentile(up_vals, 5) * up

    Q = (P - origin) @ R.T * scale          # canonical coords (m)
    length_m = float(Q[:, 0].max() - Q[:, 0].min())

    # joint info: rail positions / center / tangent / rail-top height at both ends (m)
    ends = {}
    for tag_end, idx in (('start', int(np.argmin(a_c))),
                         ('end', int(np.argmax(a_c)))):
        # end rail positions: fit a line over the 5 end slabs and extrapolate,
        # more robust than the median so a single bad slab does not skew the joint
        n_fit = min(5, len(a_c))
        if tag_end == 'start':
            sl_idx = slice(0, n_fit)
            a_e = float(a_c[0])
        else:
            sl_idx = slice(len(a_c) - n_fit, len(a_c))
            a_e = float(a_c[-1])
        if n_fit >= 3:
            s_l_e = float(np.polyval(np.polyfit(a_c[sl_idx], s_l[sl_idx], 1), a_e))
            s_r_e = float(np.polyval(np.polyfit(a_c[sl_idx], s_r[sl_idx], 1), a_e))
        else:
            s_l_e = float(np.median(s_l[sl_idx]))
            s_r_e = float(np.median(s_r[sl_idx]))
        # end rail-top height: line-fit/extrapolate over the end slabs (same as
        # lateral), more robust than a single quantile (avoids cm-level steps)
        Sm_all = best['M'] @ best['nn']
        L_all = best['M'] @ best['d']
        zc = P @ up
        z_pts, a_pts = [], []
        step = slab_w if 'slab_w' in dir() else 0.5
        rng = np.arange(a_c[0], a_c[0] + 6.0, 0.5) if tag_end == 'start' \
            else np.arange(a_c[-1] - 6.0, a_c[-1], 0.5)
        for aa in rng:
            mm = (np.abs(L_all - aa) < 0.5)
            if mm.sum() < 100:
                continue
            onr = mm & ((np.abs(Sm_all - s_l_e) < 0.25 * gauge_units) |
                        (np.abs(Sm_all - s_r_e) < 0.25 * gauge_units))
            if onr.sum() < 50:
                continue
            z_pts.append(float(np.percentile(zc[onr], 95)))
            a_pts.append(float(aa))
        if len(z_pts) >= 3:
            z_ref = float(np.polyval(np.polyfit(a_pts, z_pts, 1), a_e))
        elif z_pts:
            z_ref = float(np.median(z_pts))
        else:
            near = (np.abs(L_all - a_e) < 1.5)
            z_ref = float(np.percentile(zc[near], 95))
        # tangent: fit from the end slabs
        if len(a_c) >= 3:
            k = 2 if idx < len(a_c) // 2 else -3
            k2 = 0 if idx < len(a_c) // 2 else -1
            dA = a_c[k2] - a_c[k]
            dS = 0.5 * ((s_l[k2] - s_l[k]) + (s_r[k2] - s_r[k])) / (dA + 1e-9)
            yaw = float(np.arctan(dS))
        else:
            yaw = 0.0
        # to canonical coords (m): x = along, y = lateral (rel. track center), z = up
        cen = 0.5 * (s_l_e + s_r_e)
        ends[tag_end] = dict(
            x=(a_e - a0) * scale,
            y=(cen - s_cen0) * scale,
            z=(z_ref - np.percentile(P @ up, 5)) * scale,
            half_gauge=0.5 * (s_r_e - s_l_e) * scale,
            yaw=yaw,
            rail_l=(s_l_e - cen) * scale,
            rail_r=(s_r_e - cen) * scale,
        )

    diag = dict(M=best['M'], d=best['d'], nn=best['nn'], gauge=gauge_units,
                info=best['info'], a_c=a_c, s_l=s_l, s_r=s_r,
                votes=best['info']['votes'], Q=Q, tag=best['tag'],
                sign=best['sign'], sigma=best['sigma'],
                nvotes=best['nvotes'], ends=ends, P=P, origin=origin)
    cal = SegmentCalibration(name, R, scale, origin, up, along, cross,
                             gauge_units, None, length_m,
                             best['score'], diag)
    cal.ends = ends
    cal.nvotes = int(best['nvotes'])
    cal.sigma = float(best['sigma'])
    cal.rail_z = {k: v['z'] for k, v in ends.items()}
    # end gauges (input units), used for chained scaling by the previous segment's end gauge
    n_fit = min(5, len(a_c))
    cal.gauge_start_units = float(np.median(s_r[:n_fit] - s_l[:n_fit]))
    cal.gauge_end_units = float(np.median(s_r[-n_fit:] - s_l[-n_fit:]))
    if verbose:
        print(f'  [{name}] up "{best["tag"]}{best["sign"]:+d}"; '
              f'gauge {gauge_units:.4f} units -> scale {scale:.3f} m/unit; '
              f'{best["nvotes"]} slabs, sigma {best["sigma"]:.4f}; '
              f'length {length_m:.1f} m')
    return cal
