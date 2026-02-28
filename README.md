# Cluster Quilting — DSC 205 Project

## Overview

Applies the **Cluster Quilting** algorithm (Zheng, Chang & Allen, 2024) to discover clusters in patchwork-structured data where some samples are never jointly observed on the same features. Tested on both a real mouse facial landmark dataset and simulated Gaussian blobs.

## Datasets

**Raw** — `trialdf_24sessions.csv` (gitignored, 392MB)
- 129,603 samples x 121 features (after dropping CS valences and NaN cleaning)
- Features: pairwise distances, velocities, accelerations, areas, angles from 12 facial landmarks
- Labels: `valence` (Air_US, ITI, Sucrose_US), plus `condition`, `inj_site`, `ms_id`, `ms_n`

**Simulated** — `simulated.npz` (8MB, in repo)
- 9,999 samples x 100 features, 3 well-separated Gaussian blobs
- Cluster centers placed along orthogonal 10D subspaces (dims 0-9, 30-39, 60-69) with separation=8

## Method

**Cluster Quilting (Algorithm 1):**
1. **Patch generation** — contiguous feature blocks (1-3 per patch), overlapping samples, variable sizes, ~15% feature redundancy
2. **Patch ordering** — greedy forward search maximizing sample overlap
3. **Patchwise SVD** — rank-r truncated SVD per patch
4. **Sequential alignment** — align U vectors (sample loadings) across shared samples via least-squares
5. **k-means** — cluster on the quilted U_tilde embedding

**Key convention:** Patches have overlapping **samples** (rows) and mostly disjoint **features** (columns). Alignment uses shared samples to rotate singular vectors into a common subspace.

## Files

| File | Description |
|------|-------------|
| `spectral_quilting.ipynb` | Data cleaning, PCA, LDA, k-means baseline |
| `generate_quilt.ipynb` | Patch generation, architecture comparison, quilting pipeline, ARI sweep, 3D embeddings |
| `quilt_patches.ipynb` | Original quilting implementation (old convention — overlapping features) |
| `hyperparameter_sensitivity.ipynb` | Hyperparameter sensitivity analysis |
| `simulate_blobs.py` | Standalone script: generates 3-blob simulated data + patches |
| `simulated.npz` | Simulated dataset (data, labels, centers) |

## Patch Generation

`generate_patches()` in `generate_quilt.ipynb`:
- Tiles feature space into random contiguous blocks (min 20 cols each)
- Assigns blocks to patches with optional redundancy (shared blocks)
- Sample ranges have variable core sizes + variable overlap per boundary
- Hard minimum: 20 rows and 20 columns per patch

## Results

**ARI heatmap sweep** over n_patches={0,3,5,8,12} x overlap={0%,15%,40%,80%}:

| Config | Raw ARI | Simulated ARI |
|--------|---------|---------------|
| Full data (baseline) | 0.185 | 1.000 |
| 3p, 80% overlap (best) | 0.205 | 0.906 |
| 12p, 0% overlap (worst) | 0.027 | 0.042 |

**Key findings:**
- 0% overlap collapses ARI (~0.03) — alignment impossible without shared samples
- Fewer patches + more overlap = better (3p/80% exceeds full-data baseline on raw)
- More patches compounds alignment error; 12 patches degrades heavily
- Low-ARI configs show points collapsing to origin — unaligned patches project into null space of existing V_tilde subspace

## Plots

All in `plots/`:
- `ari_heatmap_raw.png` / `ari_heatmap_simulated.png` — ARI sweep heatmaps
- `embeddings_3d_raw.png` / `embeddings_3d_simulated.png` — 3D quilted embeddings per config
- `patch_architectures_raw.png` / `patch_architectures_simulated.png` — observation pattern grids
- `simulated_pca.png` — PCA of simulated blobs
- `lda_valence.png` — LDA projection of raw data

## Saved Patches

`patches/` (gitignored, 1.4GB total):
- `patches/raw/{3,5,8,12}p_{0,15,40,80}ov/` — raw data configs
- `patches/simulated/{3,5,8,12}p_{0,15,40,80}ov/` — simulated data configs

Regenerate with `generate_quilt.ipynb` or `simulate_blobs.py`.

## Prompt Log

| # | Date | Prompt | Notes |
|---|------|--------|-------|
| 1 | 2026-02-19 | Set up project, inspect dataset and PDF | Initial scaffold |
| 2 | 2026-02-19 | Data cleaning, StandardScaler, PCA 3D plot | Baseline visualization |
| 3 | 2026-02-19 | LDA for maximal valence separation | Supervised projection |
| 4 | 2026-02-19 | Generate patchwork missingness, save patches | Initial patch generation |
| 5 | 2026-02-28 | Fix patch convention (overlapping samples, not features), realistic patch architecture, ARI sweep heatmap, 3D embeddings, simulated blob dataset | Major overhaul of quilting pipeline |
