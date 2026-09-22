# -*- coding: utf-8 -*-
"""Join per-segment calibration results into one continuous track.

Each segment is already in its own canonical frame (m): x = along (0 = start),
y = lateral (0 = track center at start), z = up (0 ~ ground).
Stitching = place segment i+1's start on segment i's end, aligning the rails:
  - direction: end tangent of i <-> start tangent of i+1 (rotation about z)
  - lateral:   track centerlines aligned
  - height:    rail-top heights aligned
  - along:     ends touch (small overlap allowed)
"""

import numpy as np


class Placed:
    """Final placement of one segment: input coords -> global coords."""

    def __init__(self, name, R, scale, origin, M, t, cal, joint_info=None):
        self.name = name
        self.R = R              # 3x3
        self.scale = scale
        self.origin = origin    # 3
        self.M = M              # 3x3 rigid rotation in the canonical frame
        self.t = t              # 3 translation in the canonical frame
        self.cal = cal
        self.joint_info = joint_info

    def apply(self, P):
        """Input coords -> global coords (m)."""
        Q = (np.asarray(P, dtype=np.float64) - self.origin) @ self.R.T * self.scale
        return Q @ self.M.T + self.t

    def matrix4(self):
        """4x4 matrix: input coords -> global coords."""
        A = (self.R.T * self.scale) @ self.M.T      # 3x3
        b = self.t - (self.origin @ self.R.T * self.scale) @ self.M.T
        M4 = np.eye(4)
        M4[:3, :3] = A
        M4[:3, 3] = b
        return M4


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def chain(cals, overlap=0.0, verbose=True):
    """Chain calibrated segments end to end.

    Segment i placement: global = Rz(theta_i) @ Q + t_i (Q = canonical coords).
    theta from tangent continuity: yaw_start^{i+1} + theta_{i+1} = yaw_end^i + theta_i;
    t places segment i+1's start on i's end (optionally with overlap).
    """
    placed = []
    c0 = cals[0]
    placed.append(Placed(c0.name, c0.R, c0.scale, c0.origin,
                         np.eye(3), np.zeros(3), c0,
                         joint_info=dict(theta=0.0)))
    if verbose:
        print(f'  {c0.name}: start (0, 0, 0), track end x={c0.ends["end"]["x"]:.1f} m')
    for i in range(1, len(cals)):
        prev, cur = cals[i - 1], cals[i]
        P_prev = placed[-1]
        theta_prev = P_prev.joint_info['theta']
        prev_end = prev.ends['end']
        cur_start = cur.ends['start']
        # global coords and tangent at the previous segment's end
        g_end = P_prev.M @ np.array([prev_end['x'], prev_end['y'],
                                     prev_end['z']]) + P_prev.t
        yaw_global_prev = prev_end['yaw'] + theta_prev
        # align this segment's start tangent
        theta_cur = yaw_global_prev - cur_start['yaw']
        Rz = _rot_z(theta_cur)
        p_start = np.array([cur_start['x'], cur_start['y'], cur_start['z']])
        # back off along the global tangent by overlap (small overlap)
        direction = np.array([np.cos(yaw_global_prev), np.sin(yaw_global_prev), 0.0])
        t = g_end - Rz @ p_start - direction * overlap
        placed.append(Placed(cur.name, cur.R, cur.scale, cur.origin, Rz, t, cur,
                             joint_info=dict(theta=theta_cur, g_end=g_end)))
        if verbose:
            print(f'  {cur.name}: joined to {prev.name} end @ global x~{g_end[0]:.1f} m; '
                  f'rot z {np.degrees(theta_cur):+.1f} deg, '
                  f'rail-top step {cur_start["z"] - prev_end["z"]:+.3f} m, '
                  f'gauge diff {2*(cur_start["half_gauge"]-prev_end["half_gauge"]):+.3f} m')
    return placed


def refine_junctions(placed, rail_sets, window=2.5, max_dist=0.35,
                     max_shift=0.5, max_yaw=3.0, verbose=True):
    """ICP on rail points near each junction to minimize seam misalignment.

    rail_sets[i] is segment i's rail points (global coords). For each joint, run
    point-to-point ICP with the previous segment's rail as target and the current
    one as source (init = current placement), then propagate the small correction
    to this segment and all following ones.
    """
    import open3d as o3d
    n = len(placed)
    for i in range(1, n):
        gx = placed[i].joint_info['g_end'][0]
        src = rail_sets[i]
        tgt = rail_sets[i - 1]
        ms = np.abs(src[:, 0] - gx) < window
        mt = np.abs(tgt[:, 0] - gx) < window
        if ms.sum() < 200 or mt.sum() < 200:
            if verbose:
                print(f'  {placed[i].name}: too few rail points at joint, skip')
            continue
        ps = o3d.geometry.PointCloud()
        ps.points = o3d.utility.Vector3dVector(src[ms])
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(tgt[mt])
        reg = o3d.pipelines.registration.registration_icp(
            ps, pt, max_dist, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        dR = reg.transformation[:3, :3]
        dt = reg.transformation[:3, 3]
        yaw = np.degrees(np.arctan2(dR[1, 0], dR[0, 0]))
        # accept only small corrections: rails are parallel, so ICP can slide
        # onto the neighboring pair and produce metre-scale errors. Drop if too big.
        if np.linalg.norm(dt) > max_shift or abs(yaw) > max_yaw:
            if verbose:
                print(f'  {placed[i].name}: ICP correction too large (t {np.round(dt,3)} m, '
                      f'rot z {yaw:+.2f} deg), treated as slip, ignored')
            continue
        if verbose:
            print(f'  {placed[i].name}: ICP refine t {np.round(dt, 3)} m, '
                  f'rot z {yaw:+.2f} deg, rmse {reg.inlier_rmse:.4f} m (fitness {reg.fitness:.2f})')
        # propagate to segment i and all following
        for j in range(i, n):
            placed[j].M = dR @ placed[j].M
            placed[j].t = dR @ placed[j].t + dt
            if placed[j].joint_info and 'g_end' in placed[j].joint_info:
                placed[j].joint_info['g_end'] = dR @ placed[j].joint_info['g_end'] + dt
    return placed


def junction_report(placed, cals):
    """Geometric check at joints: rail position differences across each seam."""
    rows = []
    for i in range(1, len(placed)):
        a, b = cals[i - 1], cals[i]
        pa, pb = placed[i - 1], placed[i]
        # a's end rails in the global frame
        ea = a.ends['end']
        ra_l = pa.M @ np.array([ea['x'], ea['y'] + ea['rail_l'], ea['z']]) + pa.t
        ra_r = pa.M @ np.array([ea['x'], ea['y'] + ea['rail_r'], ea['z']]) + pa.t
        eb = b.ends['start']
        rb_l = pb.M @ np.array([eb['x'], eb['y'] + eb['rail_l'], eb['z']]) + pb.t
        rb_r = pb.M @ np.array([eb['x'], eb['y'] + eb['rail_r'], eb['z']]) + pb.t
        rows.append(dict(
            joint=f'{a.name}->{b.name}',
            gap_x=float(rb_l[0] - ra_l[0]),
            dy_l=float(rb_l[1] - ra_l[1]),
            dy_r=float(rb_r[1] - ra_r[1]),
            dz_l=float(rb_l[2] - ra_l[2]),
            dz_r=float(rb_r[2] - ra_r[2]),
            gauge_a=float(2 * ea['half_gauge']),
            gauge_b=float(2 * eb['half_gauge']),
        ))
    return rows


def total_extent(placed, cals):
    """Global coords of segment endpoints (track centers) -> estimated total length."""
    pts = []
    for p, c in zip(placed, cals):
        for tag in ('start', 'end'):
            e = c.ends[tag]
            pts.append(p.M @ np.array([e['x'], e['y'], e['z']]) + p.t)
    pts = np.array(pts)
    return pts
