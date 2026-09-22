# -*- coding: utf-8 -*-
"""Point-cloud I/O: PCD read, binary PLY read, PCD/PLY write.

Parsed directly with numpy to avoid open3d exploding memory on 21M points.
"""

import numpy as np


# ---------------------------------------------------------------- PCD

def read_pcd(path):
    """Read PCD v0.7 (binary or ASCII); returns (points, meta).

    points: dict field name -> numpy array (N,) or (N,count) split by FIELDS.
    Handles CloudCompare PCDs with `_` padding fields (offsets from SIZE/COUNT).
    """
    with open(path, 'rb') as f:
        raw = f.read()

    lines = []
    pos = 0
    while len(lines) < 11:
        e = raw.index(b'\n', pos)
        lines.append(raw[pos:e].decode('ascii'))
        pos = e + 1
    header_end = pos

    meta = {}
    for ln in lines:
        if ln.startswith('#') or not ln.strip():
            continue
        key, _, val = ln.partition(' ')
        meta[key] = val.strip()

    fields = meta['FIELDS'].split()
    sizes = [int(s) for s in meta['SIZE'].split()]
    types = meta['TYPE'].split()
    counts = [int(c) for c in meta['COUNT'].split()]
    n_points = int(meta['POINTS'])

    dtype_map = {'F': np.float32, 'U': np.uint8, 'I': np.int32,
                 'F4': np.float32, 'F8': np.float64}
    # some exporters use F4/F8 and may repeat counts
    dtypes = []
    for t, s in zip(types, sizes):
        if t in dtype_map:
            dtypes.append((dtype_map[t], s))
        else:  # infer from SIZE
            dtypes.append((np.float32 if s == 4 else np.uint8, s))

    offsets = []
    off = 0
    for (dt, s), c in zip(dtypes, counts):
        offsets.append(off)
        off += dt().itemsize * c
    stride = off

    data_start = header_end
    if meta.get('DATA') == 'binary':
        buf = np.frombuffer(raw, dtype=np.uint8, count=n_points * stride,
                            offset=data_start)
        buf = buf.reshape(n_points, stride)
    else:  # ascii
        txt = raw[data_start:].decode('ascii').split()
        buf = np.array(txt, dtype=np.float64).reshape(n_points, -1)
        stride = buf.shape[1] * 8

    points = {}
    for name, (dt, s), c, off_ in zip(fields, dtypes, counts, offsets):
        if c == 1:
            v = buf[:, off_:off_ + dt().itemsize].view(dt).reshape(-1)
        else:
            v = buf[:, off_:off_ + dt().itemsize * c].view(dt)
        points[name] = v
    return points, meta


def write_pcd(path, xyz, colors=None):
    """Write binary PCD v0.7 (x y z [+ rgb])."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    header = '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
    if colors is not None:
        header += 'FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
    else:
        header += 'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
    header += (f'WIDTH {len(xyz)}\nHEIGHT 1\n'
               f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\n'
               'DATA binary\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        if colors is not None:
            out = np.hstack([xyz, np.asarray(colors, dtype=np.uint8).reshape(-1, 3)])
        else:
            out = xyz
        f.write(out.astype(np.float32).tobytes())


# ---------------------------------------------------------------- PLY

def _parse_ply_header(raw):
    """Parse the PLY header; returns (vertex_byte_offset, properties, n_vertices,
    vertex_stride, binary). properties: [(name, dtype, byte_offset)]."""
    e = raw.index(b'end_header')
    hdr = raw[:e].decode('ascii', errors='ignore')
    n_vertices = 0
    fmt = 'ascii'
    props = []
    n_elems = []
    cur_elem = None
    cur_elem_n = 0
    for ln in hdr.splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == 'format':
            fmt = t[1]
        elif t[0] == 'element':
            cur_elem = t[1]
            cur_elem_n = int(t[2])
            n_elems.append((cur_elem, cur_elem_n))
        elif t[0] == 'property' and cur_elem == 'vertex':
            dt = {'float': np.float32, 'float32': np.float32,
                  'float64': np.float64, 'double': np.float64,
                  'uchar': np.uint8, 'uint8': np.uint8,
                  'int': np.int32, 'uint': np.uint32,
                  'short': np.int16, 'ushort': np.uint16}[t[1]]
            if t[1] == 'list':
                continue  # vertices usually have no list properties
            props.append((t[2], dt))
    elem_counts = {name: cnt for name, cnt in n_elems}
    n_vertices = elem_counts.get('vertex', 0)

    if fmt == 'ascii':
        return e + len(b'end_header') + 1, props, n_vertices, 0, False

    # binary: compute the vertex-data start offset by element order
    vertex_offset = e + len(b'end_header') + 1
    for name, cnt in n_elems:
        if name == 'vertex':
            break
        # skip preceding elements (e.g. face lists; vertex is usually first)
    stride = 0
    off = 0
    prop_off = []
    for name, dt in props:
        prop_off.append((name, dt, off))
        off += dt().itemsize
    stride = off
    return vertex_offset, prop_off, n_vertices, stride, True


def read_ply_xyz(path, with_colors=False):
    """Read xyz (optional rgb) from a binary PLY; returns float32 arrays.

    Parses per-vertex interleaved attributes from the declared stride; handles
    PLYs with normals/colors and other extra properties.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    voff, prop_off, n_vertices, stride, binary = _parse_ply_header(raw)
    if not binary:
        raise ValueError('ASCII PLY not supported; convert to binary with CloudCompare')

    buf = np.frombuffer(raw, dtype=np.uint8, count=n_vertices * stride,
                        offset=voff).reshape(n_vertices, stride)

    xyz = np.empty((n_vertices, 3), dtype=np.float32)
    rgb = np.empty((n_vertices, 3), dtype=np.uint8) if with_colors else None
    rgb_idx = {'red': 0, 'green': 1, 'blue': 2}
    for name, dt, off in prop_off:
        if name in ('x', 'y', 'z'):
            col = buf[:, off:off + 4].view(np.float32).reshape(-1)
            xyz[:, {'x': 0, 'y': 1, 'z': 2}[name]] = col
        elif with_colors and name in rgb_idx:
            rgb[:, rgb_idx[name]] = buf[:, off]
    return xyz, rgb


def read_ply_xyz_sample(path, target=400000, seed=0):
    """Sample-read a PLY's xyz by stride (memory-friendly, for large clouds).

    target is the desired sample size; returns (xyz float64, total point count).
    """
    with open(path, 'rb') as f:
        head = b''
        while b'end_header' not in head:
            chunk = f.read(8192)
            if not chunk:
                break
            head += chunk
    voff, prop_off, n_vertices, stride, binary = _parse_ply_header(head)
    if not binary:
        raise ValueError('ASCII PLY not supported')
    k = max(1, n_vertices // target)
    idx = np.arange(0, n_vertices, k)
    with open(path, 'rb') as f:
        xyz = np.empty((len(idx), 3), dtype=np.float64)
        cols = {}
        for name in ('x', 'y', 'z'):
            cols[name] = np.empty(len(idx), dtype=np.float32)
        # block-wise read (seek + read) to avoid loading the whole file
        block = 1 << 20
        mm = np.memmap(path, dtype=np.uint8, mode='r', offset=voff,
                       shape=(n_vertices, stride))
        s = np.asarray(mm[idx])
        for name, dt, off in prop_off:
            if name in cols:
                cols[name][:] = s[:, off:off + 4].view(np.float32).reshape(-1)
        del mm, s
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = cols['x'], cols['y'], cols['z']
    return xyz, n_vertices


def write_ply(path, xyz, colors=None, normals=None):
    """Write a binary little-endian PLY."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    has_c = colors is not None
    has_n = normals is not None
    hdr = ['ply', 'format binary_little_endian 1.0',
           f'element vertex {len(xyz)}',
           'property float x', 'property float y', 'property float z']
    if has_n:
        hdr += ['property float nx', 'property float ny', 'property float nz']
    if has_c:
        hdr += ['property uchar red', 'property uchar green',
                'property uchar blue']
    hdr += ['element face 0', 'property list uchar int vertex_indices',
            'end_header']
    with open(path, 'wb') as f:
        f.write(('\n'.join(hdr) + '\n').encode('ascii'))
        # PLY stores attributes interleaved per vertex; use align=False to
        # prevent numpy's default alignment from inserting padding
        xyz = xyz.astype(np.float32)
        fields = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        values = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        if has_n:
            nm = np.asarray(normals, np.float32).reshape(-1, 3)
            fields += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
            values += [nm[:, 0], nm[:, 1], nm[:, 2]]
        if has_c:
            cl = np.asarray(colors, np.uint8).reshape(-1, 3)
            fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            values += [cl[:, 0], cl[:, 1], cl[:, 2]]
        out = np.empty(len(xyz), dtype=np.dtype(fields, align=False))
        for (name, _), v in zip(fields, values):
            out[name] = v
        f.write(out.tobytes())
