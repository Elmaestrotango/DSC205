# Spectral Quilting — DSC 205 Project

## Overview

This project applies the **Cluster Quilting** algorithm (Zheng, Chang & Allen, 2024) to a mouse facial landmark dataset collected across 24 calcium-imaging sessions. The goal is to discover clusters among all samples even when some are never jointly observed for any feature, leveraging the patchwork structure of the data.

## Dataset

- **File:** `trialdf_24sessions.csv`
- **Rows:** ~194,000 trial observations
- **Feature columns (1–121):** pairwise distances, velocities, accelerations, areas, and angles derived from 12 facial landmarks (eye, nose, mouth, whisker regions)
- **Metadata columns:**
  - `valence` — stimulus type (ITI, Sucrose_US, Sucrose_CS, Air_CS, Air_US)
  - `condition` — behavioral state (Sated, Deprived, SatedII)
  - `inj_site` — injection site (CEM, NAC)
  - `ms_id` — unique session identifier (24 sessions)
  - `ms_n` — mouse number

## Method

The Cluster Quilting algorithm (Algorithm 1 from the paper) consists of:

1. **Patch ordering** — rank patches by overlapping signal strength (Algorithm 2: forward search)
2. **Patchwise SVD** — compute top-r SVD per data patch
3. **Sequential linear mapping** — align singular vectors across overlapping patches via least-squares
4. **k-means** — cluster on the combined, singular-value-weighted vectors

## Prompt Log

| # | Date | Prompt | Notes |
|---|------|--------|-------|
| 1 | 2026-02-19 | Set up project: create notebook and README, inspect dataset and PDF | Initial scaffold |
| 2 | 2026-02-19 | Data cleaning (drop cols >5% NaN, drop NaN rows, export npz if smaller), StandardScaler if needed, PCA top 3 PCs in 3D colored by valence (1000 pts/category) | Proof-of-concept baseline |
| 3 | 2026-02-19 | Add LDA (Linear Discriminant Analysis) to find the projection that maximally separates valence categories; 3D plot of top 3 LDs | Supervised basis change for clearest separation |
| 4 | 2026-02-19 | Generate patchwork missingness: create `generate_quilt.ipynb` with rectangular patch generation, observation-pattern heatmap, and save each patch as `.npz` | Simulated patchwork structure for Cluster Quilting |