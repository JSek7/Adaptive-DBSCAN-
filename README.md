# Adaptive DBSCAN for LiDAR Point Cloud Clustering

A clean implementation of an **adaptive-parameter DBSCAN pipeline** designed for LiDAR-based perception systems in robotics and autonomous driving.

This project explores how density-based clustering can be made robust to LiDAR’s **range-dependent point density**, where traditional global-parameter DBSCAN often fails.

---

## Motivation

Standard DBSCAN uses fixed global parameters:

- `ε` (neighbourhood radius)
- `min_samples` (minimum core density)

However, LiDAR point clouds are **non-uniform**:
- Point density decreases with distance
- Beam divergence affects spatial resolution
- 3D structure is often projected into 2D

A single global `ε` leads to:
- Fragmented far-field clusters
- Merged near-field objects
- Sensitivity to tuning

This project investigates **adaptive ε and min_samples strategies** that scale with local geometry or density statistics.

---

## High-Level Pipeline

Typical LiDAR obstacle clustering flow:

1. Region-of-interest filtering (optional)
2. Voxel grid downsampling
3. Ground removal (e.g., RANSAC plane fitting)
4. Adaptive DBSCAN clustering
5. Post-processing (centroids, bounding boxes, filtering)

---

## Repository Structure
adaptive-dbscan-core/
│
├── notebooks/ # Experimental notebooks and visualisations
├── scripts/ # Runnable pipeline entrypoints
├── src/ # Reusable implementation modules (in progress)
├── README.md
├── LICENSE
└── .gitignore


Large ROS bag files and raw LiDAR datasets are intentionally **not tracked in Git**.

---

## Current Status

Implemented / explored:

- DBSCAN clustering in 2D projected LiDAR frames
- Adaptive epsilon scaling experiments
- Cluster visualisation with bounding boxes
- RANSAC ground segmentation experiments
- Voxel downsampling utilities

In progress:

- Extraction of clean core modules into `src/`
- Formal adaptive parameter strategies
- GPU-friendly PyTorch implementation
- Synthetic benchmarking
- Unit tests

---

## Example Usage (Planned API)

```python
from adaptive_dbscan.core import adaptive_dbscan

labels = adaptive_dbscan(
    points,                    # shape (N,2) or (N,3)
    eps_mode="adaptive",
    min_samples_mode="adaptive",
    device="cuda"
)