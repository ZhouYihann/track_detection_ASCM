# Rail Top Surface Extraction

The rail head is extracted from the point cloud with the **bidirectional cloth
simulation point cloud segmentation (BCSPS)** method.

## Method

- Paper: Shi Z., Yang S., Kou R., Wang Y. *A fast railway track surface extraction
  method based on bidirectional cloth simulated point clouds.*
  Optics and Lasers in Engineering, 2024, 180: 108335.
  DOI: [10.1016/j.optlaseng.2024.108335](https://doi.org/10.1016/j.optlaseng.2024.108335)
- Original program (Windows executable only, no source code):
  <https://github.com/sz0706/BCSPS>
- Underlying cloth simulation: Zhang W., Qi J., Wan P., et al. *An easy-to-use
  airborne LiDAR data filtering method based on cloth simulation.*
  Remote Sensing, 2016, 8(6): 501. Reference implementation:
  <https://github.com/jianboqi/CSF> (Apache-2.0)


## How it works

A cloth made of uniform particles is dropped onto the point cloud and settles on
the surface below it. BCSPS applies this twice in opposite directions: from below
onto the inverted cloud, so that the cloth follows the roadbed surface `z_b`, and
then from above onto the points above the roadbed, so that it follows the upper
envelope `z_t` (mainly the rail top). A point is extracted as a rail surface point
if it lies more than `h1` above the roadbed and within the tolerance `δ` of the
upper envelope.

## Parameters

Values used for the full-length track in this repository.

| Parameter | Value | Meaning |
|---|---|---|
| `cloth_resolution` | 0.05 m | cloth grid resolution |
| `rigidness` | 3 | cloth stiffness (1/2/3 = steep slope / relief / flat) |
| `iterations` | 500 | maximum number of simulation iterations |
| `bed_open` | 0.40 m | morphological closing window applied to the rasterized terrain when building the roadbed surface; must be wider than the rail head |
| `height_diff` (`h1`) | 0.15 m | height above the roadbed at which a point counts as a raised structure |
| `top_tol` (`δ`) | 0.03 m | distance tolerance to the upper envelope |
| `z_cap` | 1.5 m | points higher than this above the roadbed are not used for the top-down simulation (removes catenary, roofs, etc.) |
| `h_tol` / `w_max` | 0.08 m / 0.45 m | rail-top ridge filter: elevation tolerance around the local rail-top height, and maximum width of a band that is kept |

## Code

```
bcsps/csf.py          cloth simulation kernel (Python port of jianboqi/CSF, numba)
bcsps/bcsps.py        bidirectional cloth simulation rail extraction
                      → track_surface_extract()
bcsps/postprocess.py  rail-top ridge filter (this repository) → rail_top_filter()
run_part1.py          run the extraction on any PLY file
validate_demo.py      accuracy check against the original authors' sample data
```

Usage:

```bash
python3 run_part1.py input.ply rail_face.ply 
```

```python
from bcsps.io_utils import read_ply_xyz
from bcsps.bcsps import track_surface_extract
from bcsps.postprocess import rail_top_filter

xyz, rgb = read_ply_xyz('input.ply', with_colors=True)
mask, _ = track_surface_extract(xyz, cloth_resolution=0.05, rigidness=3,
                                bed_open=0.40, height_diff=0.15,
                                top_tol=0.03, z_cap=1.5)
keep = rail_top_filter(xyz[mask], h_tol=0.08, w_max=0.45)
```


