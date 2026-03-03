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

**Cluster Quilting (Algorithm 1) — Sequential:**
1. **Patch generation** — contiguous feature blocks (1-3 per patch), overlapping samples, variable sizes, ~15% feature redundancy
2. **Patch ordering** — greedy forward search maximizing sample overlap
3. **Patchwise SVD** — rank-r truncated SVD per patch
4. **Sequential alignment** — align U vectors (sample loadings) across shared samples via least-squares in an O(N) chain
5. **k-means** — cluster on the quilted U_tilde embedding

**Hierarchical Variant:**
- Bottom-up pairwise merges (merge-sort style), reducing alignment chain depth from O(N) to O(log2 N)
- At each level, nodes are paired by greedy max sample overlap
- Each merge aligns the smaller node onto the larger (anchor) via lstsq on shared samples
- Isolates alignment errors per branch instead of compounding through a chain

**Key convention:** Patches have overlapping **samples** (rows) and mostly disjoint **features** (columns). Alignment uses shared samples to rotate singular vectors into a common subspace.

## Files

| File | Description |
|------|-------------|
| `spectral_quilting.ipynb` | Data cleaning, PCA, LDA, k-means baseline |
| `generate_quilt.ipynb` | Patch generation, architecture comparison, quilting pipeline, ARI sweep, 3D embeddings |
| `quilt_patches.ipynb` | Original quilting implementation (old convention — overlapping features) |
| `hyperparameter_sensitivity.ipynb` | Hyperparameter sensitivity analysis |
| `hierarchical_quilting.py` | Sequential vs hierarchical quilting comparison (5 replicates, 1–15 patches, paired stats) |
| `simulate_blobs.py` | Standalone script: generates 3-blob simulated data + patches |
| `simulated.npz` | Simulated dataset (data, labels, centers) |

## Patch Generation

`generate_patches()` in `generate_quilt.ipynb`:
- Tiles feature space into random contiguous blocks (min 20 cols each)
- Assigns blocks to patches with optional redundancy (shared blocks)
- Sample ranges have variable core sizes + variable overlap per boundary
- Hard minimum: 20 rows and 20 columns per patch

## Results

### Sequential quilting (from `generate_quilt.ipynb`)

**ARI heatmap sweep** over n_patches={0,3,5,8,12} x overlap={0%,15%,40%,80%}:

| Config | Raw ARI | Simulated ARI |
|--------|---------|---------------|
| Full data (baseline) | 0.185 | 1.000 |
| 3p, 80% overlap (best) | 0.205 | 0.906 |
| 12p, 0% overlap (worst) | 0.027 | 0.042 |

**Key findings (sequential):**
- 0% overlap collapses ARI (~0.03) — alignment impossible without shared samples
- Fewer patches + more overlap = better (3p/80% exceeds full-data baseline on raw)
- More patches compounds alignment error; 12 patches degrades heavily
- Low-ARI configs show points collapsing to origin — unaligned patches project into null space of existing V_tilde subspace

### Sequential vs hierarchical (from `hierarchical_quilting.py`)

**Replicated comparison sweep** over n_patches=1–15 x overlap={0%,10%,20%,40%,80%}, 5 replicates per config (different random seeds for patch geometry). Paired t-tests with Bonferroni correction at each grid point.

**Raw data — representative mean ARI values:**

| Config | Sequential | Hierarchical | Diff (H-S) |
|--------|-----------|-------------|------------|
| Full data (baseline) | 0.185 | 0.185 | 0.000 |
| 1p (any overlap) | 0.205 | 0.205 | 0.000 |
| 3p, 10% overlap | 0.122 | 0.140 | +0.018 |
| 7p, 20% overlap | 0.160 | 0.084 | -0.076 |
| 12p, 20% overlap | 0.137 | 0.028 | -0.110 |
| 14p, 40% overlap | 0.128 | 0.024 | -0.104 |

**Key findings:**
- Sequential consistently outperforms hierarchical from ~6 patches onward, across all non-zero overlap levels
- Hierarchical deficit peaks at 10–20% overlap with 7–14 patches (ARI gap of -0.05 to -0.11 on raw data)
- At very low patch counts (2-3), hierarchical is comparable or slightly better (tree depth is only 1-2 levels, so the global separation problem hasn't emerged)
- At 80% overlap the gap narrows — abundant shared samples partially compensate for suboptimal pairing
- Both methods are identical at 0% overlap (no alignment possible) and for n_patches=1 (no alignment needed)
- Hierarchical shows higher variance across replicates, indicating greater sensitivity to specific patch geometry

**Failure mode analysis:**
- *Sequential*: alignment errors compound along the O(N) chain — later patches align against an increasingly noisy accumulated embedding. Early bad merges are most damaging (longest propagation tail)
- *Hierarchical*: greedy pairing pulls similar (adjacent) patches into the same branch, leaving the final tree merge to bridge two geographically separated halves with minimal shared samples. This "local greediness → global separation" effect worsens with more patches and deeper trees

**Simulated data:**
- More variable than raw — hierarchical occasionally wins at specific configs (e.g., 6p/80%: +0.210, 6p/40%: +0.153) but also has catastrophic failures (8p/10%: -0.208, 9p/40%: -0.181)
- The well-separated cluster structure means alignment quality matters more (errors can flip cluster assignments entirely), amplifying both successes and failures

**Statistical significance:**
- 0/75 grid points significant after Bonferroni correction (alpha=0.00067) on either dataset
- Several raw data configs approach uncorrected significance (12p/20%: p=0.003, 14p/40%: p=0.008)
- Limited power with n=5 replicates (df=4); directional trends are consistent but require more replicates for formal confirmation

## Plots

All in `plots/`:

**Baselines** (from `generate_quilt.ipynb` / `spectral_quilting.ipynb`):
- `simulated_pca.png` — PCA of simulated blobs
- `lda_valence.png` — LDA projection of raw data
- `patch_architectures_raw.png` / `patch_architectures_simulated.png` — observation pattern grids

**Sequential quilting sweep** (from `generate_quilt.ipynb`):
- `ari_heatmap_raw.png` / `ari_heatmap_simulated.png` — ARI sweep heatmaps
- `embeddings_3d_raw.png` / `embeddings_3d_simulated.png` — 3D quilted embeddings per config

**Sequential vs hierarchical comparison** (from `hierarchical_quilting.py`, 5 replicates):
- `heatmap_comparison_raw.png` / `heatmap_comparison_simulated.png` — side-by-side mean ARI heatmaps (16x5 grid, +/-std annotations)
- `ari_line_comparison.png` — ARI vs n_patches line plots with SEM error bands (2 rows x 5 overlap levels)
- `effect_size_raw.png` / `effect_size_sim.png` — mean(hierarchical - sequential) ARI difference heatmaps with diverging colormap
- `embeddings_3d_seq_raw.png` / `embeddings_3d_hier_raw.png` — 3D scatter grids, sequential vs hierarchical, raw data (seed 0)
- `embeddings_3d_seq_sim.png` / `embeddings_3d_hier_sim.png` — 3D scatter grids, sequential vs hierarchical, simulated data (seed 0)

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
| 6 | 2026-02-28 | Hierarchical quilting: bottom-up pairwise merge variant, comparison sweep across grid, side-by-side heatmaps and line plots | hierarchical_quilting.py |
| 7 | 2026-03-02 | Expanded comparison: 5 replicates, 1–15 patches, 5 overlap levels, SEM error bands, paired t-tests with Bonferroni, effect size heatmaps, 3D embedding grids | hierarchical_quilting.py overhaul |
