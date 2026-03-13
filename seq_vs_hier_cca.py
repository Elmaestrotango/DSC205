#!/usr/bin/env python3
"""
seq_vs_hier_cca.py

Compare sequential vs hierarchical alignment (both lstsq and CCA)
across 20, 50, and 100 patches on the simulated dataset.

For each configuration, runs 5 seeds and reports:
  - ARI (k-means on quilted embedding vs ground truth)
  - Column-info (mean R² of regressing all columns onto the embedding)

Outputs plots to plots/seqvshierCCA/.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

from hierarchical_quilting import generate_patches, greedy_patch_ordering, _greedy_pairing

# ── Config ──────────────────────────────────────────────────────────
CONFIGS = [
    {'n_patches': 20,  'overlap_frac': 0.40},
    {'n_patches': 50,  'overlap_frac': 0.60},
    {'n_patches': 100, 'overlap_frac': 0.80},
]
SEEDS = [42, 123, 456, 789, 1024]
RANK = 3
OUT_DIR = Path('plots/seqvshierCCA')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load simulated data ─────────────────────────────────────────────
sim = np.load('simulated.npz')
X_full = sim['data']
labels_true = sim['labels']
N_total, N_cols = X_full.shape
K = len(np.unique(labels_true))


# ── Helpers ──────────────────────────────────────────────────────────
def _mat_sqrt_inv(C):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs * (1.0 / np.sqrt(eigvals)) @ eigvecs.T


def column_info(U_node, X_sub):
    """Mean R² across all columns when regressing X_sub onto U_node."""
    B, _, _, _ = np.linalg.lstsq(U_node, X_sub, rcond=None)
    X_hat = U_node @ B
    ss_res = np.sum((X_sub - X_hat) ** 2, axis=0)
    ss_tot = np.sum((X_sub - X_sub.mean(axis=0)) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2 = np.zeros(X_sub.shape[1])
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    r2 = np.clip(r2, 0.0, 1.0)
    return float(r2.mean())


# ── Baseline ─────────────────────────────────────────────────────────
X_pca = PCA(n_components=RANK).fit_transform(X_full)
baseline_ci = column_info(X_pca, X_full)
print(f"Baseline column-info (full PCA, rank {RANK}): {baseline_ci:.4f}")


# ── Sequential CCA ──────────────────────────────────────────────────
def sequential_cca(patches_data, X_full, r):
    M = X_full.shape[0]
    ordering = greedy_patch_ordering(patches_data, M=M)
    U_tilde = np.zeros((M, r))
    covered = np.zeros(M, dtype=bool)
    REG = 1e-8

    p0 = patches_data[ordering[0]]
    U, S, Vt = np.linalg.svd(X_full[np.ix_(p0['row_idx'], p0['col_idx'])],
                              full_matrices=False)
    U_tilde[p0['row_idx']] = U[:, :r]
    covered[p0['row_idx']] = True

    for step in range(1, len(ordering)):
        m = ordering[step]
        p = patches_data[m]
        row_idx, col_idx = p['row_idx'], p['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)],
                                  full_matrices=False)
        U_r = U[:, :r]

        is_cov = covered[row_idx]
        ov_global = row_idx[is_cov]
        new_global = row_idx[~is_cov]
        n_ov = len(ov_global)

        if n_ov < r:
            U_tilde[row_idx] = U_r
            covered[row_idx] = True
            continue

        ov_local = np.searchsorted(row_idx, ov_global)
        covered_idx = np.where(covered)[0]

        U_a_ov = U_tilde[ov_global]
        U_b_ov = U_r[ov_local]
        mu_a, mu_b = U_a_ov.mean(0), U_b_ov.mean(0)
        A_c, B_c = U_a_ov - mu_a, U_b_ov - mu_b

        C_aa = A_c.T @ A_c / (n_ov - 1) + REG * np.eye(r)
        C_bb = B_c.T @ B_c / (n_ov - 1) + REG * np.eye(r)
        C_ab = A_c.T @ B_c / (n_ov - 1)

        C_aa_isq = _mat_sqrt_inv(C_aa)
        C_bb_isq = _mat_sqrt_inv(C_bb)
        M_cca = C_aa_isq @ C_ab @ C_bb_isq
        P, s, Qt = np.linalg.svd(M_cca, full_matrices=False)
        W_a = C_aa_isq @ P
        W_b = C_bb_isq @ Qt.T

        U_a_cca = (U_tilde[covered_idx] - mu_a) @ W_a
        U_b_cca = (U_r - mu_b) @ W_b

        ov_in_covered = np.searchsorted(covered_idx, ov_global)
        U_tilde[covered_idx] = U_a_cca
        U_tilde[ov_global] = 0.5 * (U_a_cca[ov_in_covered] + U_b_cca[ov_local])
        if len(new_global) > 0:
            new_local = np.searchsorted(row_idx, new_global)
            U_tilde[new_global] = U_b_cca[new_local]

        covered[row_idx] = True

    return U_tilde


# ── Hierarchical CCA ────────────────────────────────────────────────
def hierarchical_cca(patches_data, X_full, r):
    N = X_full.shape[0]
    REG = 1e-8

    # Init leaf nodes
    nodes = []
    for p in patches_data:
        row_idx, col_idx = p['row_idx'], p['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)],
                                  full_matrices=False)
        nodes.append({
            'U_dense': U[:, :r].copy(),
            'row_list': np.asarray(row_idx, dtype=np.int64),
        })

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes, N)
        next_level = []

        for i, j in pairs:
            nd_a, nd_b = nodes[i], nodes[j]
            rows_a, rows_b = nd_a['row_list'], nd_b['row_list']
            overlap = np.intersect1d(rows_a, rows_b, assume_unique=True)
            n_ov = len(overlap)

            if n_ov >= r:
                ov_a = np.searchsorted(rows_a, overlap)
                ov_b = np.searchsorted(rows_b, overlap)

                U_a_ov = nd_a['U_dense'][ov_a]
                U_b_ov = nd_b['U_dense'][ov_b]
                mu_a, mu_b = U_a_ov.mean(0), U_b_ov.mean(0)
                A_c, B_c = U_a_ov - mu_a, U_b_ov - mu_b

                C_aa = A_c.T @ A_c / (n_ov - 1) + REG * np.eye(r)
                C_bb = B_c.T @ B_c / (n_ov - 1) + REG * np.eye(r)
                C_ab = A_c.T @ B_c / (n_ov - 1)

                C_aa_isq = _mat_sqrt_inv(C_aa)
                C_bb_isq = _mat_sqrt_inv(C_bb)
                M_cca = C_aa_isq @ C_ab @ C_bb_isq
                P, s, Qt = np.linalg.svd(M_cca, full_matrices=False)
                W_a = C_aa_isq @ P
                W_b = C_bb_isq @ Qt.T

                U_a_cca = (nd_a['U_dense'] - mu_a) @ W_a
                U_b_cca = (nd_b['U_dense'] - mu_b) @ W_b
            else:
                U_a_cca = nd_a['U_dense']
                U_b_cca = nd_b['U_dense']
                ov_a = np.searchsorted(rows_a, overlap)
                ov_b = np.searchsorted(rows_b, overlap)

            # Merge
            new_in_b = np.setdiff1d(rows_b, rows_a, assume_unique=True)
            n_a, n_new_b = len(rows_a), len(new_in_b)
            U_cat = np.empty((n_a + n_new_b, r))
            U_cat[:n_a] = U_a_cca
            if n_ov >= r:
                U_cat[ov_a] = 0.5 * (U_a_cca[ov_a] + U_b_cca[ov_b])
            if n_new_b > 0:
                U_cat[n_a:] = U_b_cca[np.searchsorted(rows_b, new_in_b)]

            union = np.concatenate([rows_a, new_in_b])
            order = np.argsort(union)
            next_level.append({'U_dense': U_cat[order], 'row_list': union[order]})

        if unpaired is not None:
            next_level.append(nodes[unpaired])
        nodes = next_level

    # Extract final embedding
    final = nodes[0]
    U_tilde = np.zeros((N, r))
    U_tilde[final['row_list']] = final['U_dense']
    return U_tilde


# ── Sequential lstsq ─────────────────────────────────────────────────
def sequential_lstsq(patches_data, X_full, r):
    M = X_full.shape[0]
    ordering = greedy_patch_ordering(patches_data, M=M)
    U_tilde = np.zeros((M, r))
    covered = np.zeros(M, dtype=bool)

    p0 = patches_data[ordering[0]]
    U, S, Vt = np.linalg.svd(X_full[np.ix_(p0['row_idx'], p0['col_idx'])],
                              full_matrices=False)
    U_tilde[p0['row_idx']] = U[:, :r]
    covered[p0['row_idx']] = True

    for step in range(1, len(ordering)):
        m = ordering[step]
        p = patches_data[m]
        row_idx, col_idx = p['row_idx'], p['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)],
                                  full_matrices=False)
        U_r = U[:, :r]

        is_cov = covered[row_idx]
        ov_global = row_idx[is_cov]
        new_global = row_idx[~is_cov]
        n_ov = len(ov_global)

        if n_ov < r:
            U_tilde[row_idx] = U_r
            covered[row_idx] = True
            continue

        ov_local = np.searchsorted(row_idx, ov_global)
        G, _, _, _ = np.linalg.lstsq(U_r[ov_local], U_tilde[ov_global],
                                     rcond=None)
        if len(new_global) > 0:
            new_local = np.searchsorted(row_idx, new_global)
            U_tilde[new_global] = U_r[new_local] @ G

        covered[row_idx] = True

    return U_tilde


# ── Hierarchical lstsq ──────────────────────────────────────────────
def hierarchical_lstsq(patches_data, X_full, r):
    N = X_full.shape[0]

    nodes = []
    for p in patches_data:
        row_idx, col_idx = p['row_idx'], p['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)],
                                  full_matrices=False)
        nodes.append({
            'U_dense': U[:, :r].copy(),
            'row_list': np.asarray(row_idx, dtype=np.int64),
        })

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes, N)
        next_level = []

        for i, j in pairs:
            if len(nodes[i]['row_list']) >= len(nodes[j]['row_list']):
                anchor, child = nodes[i], nodes[j]
            else:
                anchor, child = nodes[j], nodes[i]

            rows_a, rows_b = anchor['row_list'], child['row_list']
            overlap = np.intersect1d(rows_a, rows_b, assume_unique=True)
            n_ov = len(overlap)

            if n_ov >= r:
                ov_a = np.searchsorted(rows_a, overlap)
                ov_b = np.searchsorted(rows_b, overlap)
                G, _, _, _ = np.linalg.lstsq(
                    child['U_dense'][ov_b], anchor['U_dense'][ov_a], rcond=None)
                U_b_aligned = child['U_dense'] @ G
            else:
                U_b_aligned = child['U_dense']

            new_in_b = np.setdiff1d(rows_b, rows_a, assume_unique=True)
            n_a, n_new = len(rows_a), len(new_in_b)
            U_cat = np.empty((n_a + n_new, r))
            U_cat[:n_a] = anchor['U_dense']
            if n_new > 0:
                U_cat[n_a:] = U_b_aligned[np.searchsorted(rows_b, new_in_b)]

            union = np.concatenate([rows_a, new_in_b])
            order = np.argsort(union)
            next_level.append({'U_dense': U_cat[order], 'row_list': union[order]})

        if unpaired is not None:
            next_level.append(nodes[unpaired])
        nodes = next_level

    final = nodes[0]
    U_tilde = np.zeros((N, r))
    U_tilde[final['row_list']] = final['U_dense']
    return U_tilde


# ── Run experiments ──────────────────────────────────────────────────
results = []

for cfg in CONFIGS:
    n_p = cfg['n_patches']
    ov_f = cfg['overlap_frac']
    print(f"\n{'='*60}")
    print(f"  {n_p} patches, {ov_f:.0%} overlap")
    print(f"{'='*60}")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        try:
            patches = generate_patches((N_total, N_cols), n_p,
                                       overlap_frac=ov_f, rng=rng)
        except Exception as e:
            print(f"  Seed {seed}: patch generation failed ({e}), skipping")
            continue

        # Sequential CCA
        U_seq_cca = sequential_cca(patches, X_full, RANK)
        km = KMeans(n_clusters=K, n_init=20, random_state=42)
        ari_seq_cca = adjusted_rand_score(labels_true, km.fit_predict(U_seq_cca))
        ci_seq_cca = column_info(U_seq_cca, X_full) / baseline_ci

        # Hierarchical CCA
        U_hier_cca = hierarchical_cca(patches, X_full, RANK)
        km = KMeans(n_clusters=K, n_init=20, random_state=42)
        ari_hier_cca = adjusted_rand_score(labels_true, km.fit_predict(U_hier_cca))
        ci_hier_cca = column_info(U_hier_cca, X_full) / baseline_ci

        # Sequential lstsq
        U_seq_ls = sequential_lstsq(patches, X_full, RANK)
        km = KMeans(n_clusters=K, n_init=20, random_state=42)
        ari_seq_ls = adjusted_rand_score(labels_true, km.fit_predict(U_seq_ls))
        ci_seq_ls = column_info(U_seq_ls, X_full) / baseline_ci

        # Hierarchical lstsq
        U_hier_ls = hierarchical_lstsq(patches, X_full, RANK)
        km = KMeans(n_clusters=K, n_init=20, random_state=42)
        ari_hier_ls = adjusted_rand_score(labels_true, km.fit_predict(U_hier_ls))
        ci_hier_ls = column_info(U_hier_ls, X_full) / baseline_ci

        results.append({
            'n_patches': n_p, 'overlap_frac': ov_f, 'seed': seed,
            'ari_seq_cca': ari_seq_cca, 'ari_hier_cca': ari_hier_cca,
            'ari_seq_ls': ari_seq_ls, 'ari_hier_ls': ari_hier_ls,
            'ci_seq_cca': ci_seq_cca, 'ci_hier_cca': ci_hier_cca,
            'ci_seq_ls': ci_seq_ls, 'ci_hier_ls': ci_hier_ls,
        })
        print(f"  Seed {seed}:  "
              f"SeqCCA={ari_seq_cca:.3f}  HierCCA={ari_hier_cca:.3f}  "
              f"SeqLS={ari_seq_ls:.3f}  HierLS={ari_hier_ls:.3f}")


# ── Aggregate ────────────────────────────────────────────────────────
METHODS = ['seq_cca', 'hier_cca', 'seq_ls', 'hier_ls']
METHOD_LABELS = {
    'seq_cca': 'Seq CCA', 'hier_cca': 'Hier CCA',
    'seq_ls': 'Seq lstsq', 'hier_ls': 'Hier lstsq',
}
METHOD_COLORS = {
    'seq_cca': '#ff7f00', 'hier_cca': '#4daf4a',
    'seq_ls': '#e41a1c', 'hier_ls': '#377eb8',
}

print("\n\n=== Summary ===")
for cfg in CONFIGS:
    n_p = cfg['n_patches']
    subset = [r for r in results if r['n_patches'] == n_p]
    if not subset:
        print(f"  {n_p} patches: no valid results")
        continue
    print(f"  {n_p:3d} patches ({cfg['overlap_frac']:.0%} ov):")
    for m in METHODS:
        ari = np.array([r[f'ari_{m}'] for r in subset])
        ci = np.array([r[f'ci_{m}'] for r in subset])
        print(f"    {METHOD_LABELS[m]:12s}: ARI={ari.mean():.3f}+/-{ari.std():.3f}  "
              f"CI={ci.mean():.3f}+/-{ci.std():.3f}")


# ================================================================
# PLOTS — 4-bar grouped comparison
# ================================================================
config_labels = [f"{c['n_patches']}p / {c['overlap_frac']:.0%} ov" for c in CONFIGS]
n_cfgs = len(CONFIGS)
n_methods = len(METHODS)
width = 0.18  # bar width — 4 bars fit with some gap

def _grouped_bar_plot(metric_key, ylabel, title_suffix, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_cfgs)

    for mi, m in enumerate(METHODS):
        means, stds, pts = [], [], []
        for cfg in CONFIGS:
            subset = [r for r in results if r['n_patches'] == cfg['n_patches']]
            vals = np.array([r[f'{metric_key}_{m}'] for r in subset])
            means.append(vals.mean()); stds.append(vals.std()); pts.append(vals)

        offset = (mi - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=METHOD_LABELS[m], color=METHOD_COLORS[m],
                      edgecolor='k', alpha=0.85)

        # Overlay seed points
        for i in range(n_cfgs):
            jitter = np.random.default_rng(mi).uniform(-0.04, 0.04, len(pts[i]))
            ax.scatter(x[i] + offset + jitter, pts[i], color='k', s=20,
                       zorder=5, alpha=0.6, edgecolors='white', linewidths=0.4)

        # Annotate means
        for bar, mu in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{mu:.3f}', ha='center', va='bottom', fontweight='bold',
                    fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Sequential vs Hierarchical (CCA & lstsq) — {title_suffix}\n'
                 f'(Simulated data, 5 seeds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT_DIR / filename}")


_grouped_bar_plot('ari', 'Adjusted Rand Index', 'ARI', 'ari_comparison.png')
_grouped_bar_plot('ci', 'Column Info (fraction of baseline)', 'Column Info',
                  'column_info_comparison.png')

# ── Plot 3: Difference plot (CCA hier-seq vs lstsq hier-seq) ────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, metric, ylabel, title in [
    (ax1, 'ari', 'Difference (Hier - Seq)', 'ARI: Hierarchical - Sequential'),
    (ax2, 'ci', 'Difference (Hier - Seq)', 'Column Info: Hierarchical - Sequential'),
]:
    for i, cfg in enumerate(CONFIGS):
        subset = [r for r in results if r['n_patches'] == cfg['n_patches']]
        # CCA diffs
        diffs_cca = np.array([r[f'{metric}_hier_cca'] - r[f'{metric}_seq_cca']
                              for r in subset])
        jitter = np.random.default_rng(i).uniform(-0.06, 0.06, len(diffs_cca))
        ax.scatter(np.full(len(diffs_cca), i) - 0.12 + jitter, diffs_cca,
                   s=45, zorder=5, alpha=0.8, edgecolors='k', linewidths=0.5,
                   color=METHOD_COLORS['hier_cca'], label='CCA' if i == 0 else '')
        ax.errorbar(i - 0.12, diffs_cca.mean(), yerr=diffs_cca.std(),
                    fmt='D', color=METHOD_COLORS['hier_cca'],
                    markersize=7, capsize=5, zorder=6, linewidth=2,
                    markeredgecolor='k')
        # lstsq diffs
        diffs_ls = np.array([r[f'{metric}_hier_ls'] - r[f'{metric}_seq_ls']
                             for r in subset])
        jitter = np.random.default_rng(i + 10).uniform(-0.06, 0.06, len(diffs_ls))
        ax.scatter(np.full(len(diffs_ls), i) + 0.12 + jitter, diffs_ls,
                   s=45, zorder=5, alpha=0.8, edgecolors='k', linewidths=0.5,
                   color=METHOD_COLORS['hier_ls'], label='lstsq' if i == 0 else '')
        ax.errorbar(i + 0.12, diffs_ls.mean(), yerr=diffs_ls.std(),
                    fmt='D', color=METHOD_COLORS['hier_ls'],
                    markersize=7, capsize=5, zorder=6, linewidth=2,
                    markeredgecolor='k')

    ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(range(n_cfgs))
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Hierarchical vs Sequential: CCA and lstsq\n'
             '(> 0 means hierarchical wins)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT_DIR / 'difference_plot.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {OUT_DIR / 'difference_plot.png'}")

print("\nDone.")
