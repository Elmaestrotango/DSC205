"""Generate 3 well-separated Gaussian blobs in 100D and produce patchwork patches.

Outputs:
  - ~/DSC205/simulated.npz          — full simulated dataset
  - patches/simulated/patch_*.npz   — patchwork patches (disjoint features, overlapping samples)
  - plots/simulated_pca.png         — PCA visualization colored by cluster
  - plots/simulated_patches.png     — patchwork observation pattern
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
N_SAMPLES = 10_000
N_FEATURES = 100
N_CLUSTERS = 3
SEED = 42

# Cluster centers: well-separated in 100D
# Place centers along orthogonal directions with large separation
rng = np.random.default_rng(SEED)
centers = np.zeros((N_CLUSTERS, N_FEATURES))
centers[0, :10] = 8.0    # cluster 0 lives in first 10 dims
centers[1, 30:40] = 8.0  # cluster 1 lives in dims 30–39
centers[2, 60:70] = 8.0  # cluster 2 lives in dims 60–69

# ── Generate blobs ─────────────────────────────────────────────────────
samples_per = N_SAMPLES // N_CLUSTERS
labels = np.repeat(np.arange(N_CLUSTERS), samples_per)
X = np.vstack([
    rng.normal(loc=centers[k], scale=1.0, size=(samples_per, N_FEATURES))
    for k in range(N_CLUSTERS)
])

# Shuffle
perm = rng.permutation(len(labels))
X = X[perm]
labels = labels[perm]

print(f"Generated data: {X.shape[0]} samples x {X.shape[1]} features")
print(f"Cluster counts: {np.bincount(labels)}")

# ── Standardize ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Save full dataset ──────────────────────────────────────────────────
out_path = Path(__file__).parent / "simulated.npz"
np.savez(out_path, data=X_scaled, labels=labels, centers=centers)
print(f"Saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

# ── PCA visualization ─────────────────────────────────────────────────
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

plots_dir = Path(__file__).parent / "plots"
plots_dir.mkdir(exist_ok=True)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
colors = ["#e41a1c", "#377eb8", "#4daf4a"]
for k in range(N_CLUSTERS):
    mask = labels == k
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
        s=3, alpha=0.3, label=f"Cluster {k}", color=colors[k],
    )
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
ax.set_title("Simulated 3-blob dataset — PCA")
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig(plots_dir / "simulated_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {plots_dir / 'simulated_pca.png'}")

# ── Patch generation (reuse same logic as generate_quilt.ipynb) ────────
def generate_patches(shape, n_patches, overlap_frac=0.25):
    M, N = shape
    base_width = N // n_patches
    remainder = N % n_patches
    col_sizes = [base_width + (1 if i < remainder else 0) for i in range(n_patches)]
    col_breaks = np.concatenate([[0], np.cumsum(col_sizes)])

    base_rows = M // n_patches
    overlap_size = int(base_rows * overlap_frac)

    patches = []
    for i in range(n_patches):
        row_start = i * base_rows
        row_end = (i + 1) * base_rows if i < n_patches - 1 else M
        if i > 0:
            row_start -= overlap_size
        if i < n_patches - 1:
            row_end += overlap_size
        row_start = max(0, row_start)
        row_end = min(M, row_end)
        patches.append({
            "row_idx": np.arange(row_start, row_end),
            "col_idx": np.arange(col_breaks[i], col_breaks[i + 1]),
        })
    return patches


M, N = X_scaled.shape
n_patches = 3
patches = generate_patches((M, N), n_patches=n_patches, overlap_frac=0.25)

print(f"\nGenerated {n_patches} patches (disjoint features, overlapping samples):")
for i, p in enumerate(patches):
    r, c = p["row_idx"], p["col_idx"]
    print(f"  Patch {i}: rows [{r[0]}–{r[-1]}] ({len(r)}), "
          f"cols [{c[0]}–{c[-1]}] ({len(c)})")
for i in range(len(patches) - 1):
    overlap = len(set(patches[i]["row_idx"]) & set(patches[i + 1]["row_idx"]))
    print(f"  Overlap {i}<->{i+1}: {overlap} shared samples")

# Save patches
patch_dir = Path(__file__).parent / "patches" / "simulated"
patch_dir.mkdir(parents=True, exist_ok=True)

for i, p in enumerate(patches):
    row_idx, col_idx = p["row_idx"], p["col_idx"]
    data = X_scaled[np.ix_(row_idx, col_idx)]
    fname = patch_dir / f"patch_{i:02d}.npz"
    np.savez(fname, data=data, row_idx=row_idx, col_idx=col_idx,
             labels=labels[row_idx])
    print(f"Saved {fname}  shape={data.shape}")

# Patchwork visualization
mask = np.zeros((M, N), dtype=np.int8)
for p in patches:
    mask[np.ix_(p["row_idx"], p["col_idx"])] = 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(mask, aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
ax.set_title(f"Simulated patchwork — {n_patches} patches, obs rate: {mask.mean():.1%}")
ax.set_xlabel("Feature index")
ax.set_ylabel("Sample index")
plt.tight_layout()
plt.savefig(plots_dir / "simulated_patches.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {plots_dir / 'simulated_patches.png'}")
