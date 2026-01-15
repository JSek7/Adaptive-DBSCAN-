# Adaptive DBSCAN (LiDAR Clustering Core)

Core implementation of an adaptive-parameter DBSCAN pipeline for LiDAR point cloud
clustering (robotics / autonomous driving use-cases).

This repository is intended to hold the reusable algorithm core (clean Python
modules). Experimental notebooks and quick tests live in `examples/`.

---------------------------------------------------------------------

OVERVIEW

This project focuses on density-based clustering of LiDAR point clouds using an
adaptive variant of DBSCAN. Instead of relying on a single global epsilon (ε) and
min_samples, parameters are adapted based on local point density and neighbourhood
statistics.

The goal is to produce a clustering core that is:
- robust to varying point density
- suitable for LiDAR perception pipelines
- easy to integrate into robotics or autonomy stacks

---------------------------------------------------------------------

FEATURES

- Adaptive tuning of DBSCAN parameters (ε and/or min_samples)
- Designed for LiDAR point clouds in 2D or 3D
- Compatible with typical preprocessing steps (voxelisation, ground removal)
- Intended to be GPU-friendly (PyTorch-based implementation planned)

---------------------------------------------------------------------

REPOSITORY STRUCTURE

adaptive-dbscan-core/
  adaptive_dbscan/
    __init__.py
    core.py            # adaptive DBSCAN implementation (WIP)
    adaptive_eps.py    # adaptive epsilon logic (WIP)
    utils.py           # helper functions (WIP)

  examples/
    dbscan-test.ipynb
    test.ipynb

  README.md

---------------------------------------------------------------------

INSTALLATION / CLONE

Clone the repository locally:

git clone https://github.com/JSek7/Adaptive-DBSCAN-.git
cd Adaptive-DBSCAN-

Note: the repository name may be renamed later to `adaptive-dbscan-core`. GitHub
will automatically redirect old links.

---------------------------------------------------------------------

PLANNED USAGE API

Example of the intended interface (subject to change as the implementation is
extracted from notebooks):

from adaptive_dbscan.core import adaptive_dbscan

labels = adaptive_dbscan(
    points,                   # shape (N, 2) or (N, 3)
    device="cuda",            # "cpu" or "cuda"
    eps_mode="adaptive",      # "adaptive" or "fixed"
    min_samples_mode="adaptive"
)

---------------------------------------------------------------------

RECOMMENDED LIDAR PIPELINE

Typical flow for LiDAR obstacle clustering:

1. Optional ROI crop or range filtering
2. Voxel grid downsampling
3. Ground removal (RANSAC plane segmentation)
4. Adaptive DBSCAN on non-ground points

---------------------------------------------------------------------

ROADMAP

- Extract core DBSCAN logic from notebooks into adaptive_dbscan/core.py
- Implement adaptive epsilon and min_samples strategies
- Add synthetic unit tests and simple benchmarks
- Add example on a real LiDAR frame
- Document integration into a perception pipeline
- Add optional ground removal utilities

---------------------------------------------------------------------

LICENSE

MIT license 
