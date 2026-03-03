#!/usr/bin/env python3
"""Hierarchical Cluster Quilting — bottom-up pairwise merge variant.

Compares the original sequential (chain) quilting algorithm against a
hierarchical (merge-sort style) variant that reduces alignment chain depth
from O(N) to O(log2 N).

Runs triplicate experiments across a fine grid (1-15 patches x 5 overlap
levels) with paired statistical tests and error-band visualizations.

Usage:
    python hierarchical_quilting.py

Outputs:
    plots/heatmap_comparison_raw.png
    plots/heatmap_comparison_simulated.png
    plots/ari_line_comparison.png
    plots/embeddings_3d_seq_raw.png
    plots/embeddings_3d_hier_raw.png
    plots/embeddings_3d_seq_sim.png
    plots/embeddings_3d_hier_sim.png
    plots/effect_size_raw.png
    plots/effect_size_sim.png
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────
MIN_PATCH_ROWS = 20
MIN_PATCH_COLS = 20


# ── Patch generation (from generate_quilt.ipynb cell-3) ───────────────────

def generate_patches(shape, n_patches, overlap_frac=0.25,
                     feature_redundancy=0.15, max_blocks_per_patch=3, rng=None):
    """Generate patchwork patches with contiguous feature blocks."""
    if rng is None:
        rng = np.random.default_rng(42)

    M, N = shape

    # Feature blocks
    block_pool = []
    pos = 0
    while pos < N:
        width = rng.integers(MIN_PATCH_COLS,
                             max(MIN_PATCH_COLS + 1, N // n_patches + 10))
        end = min(N, pos + width)
        if N - end < MIN_PATCH_COLS and end < N:
            end = N
        block_pool.append((pos, end))
        pos = end

    rng.shuffle(block_pool)
    patch_blocks = [[] for _ in range(n_patches)]
    for p in range(n_patches):
        patch_blocks[p].append(block_pool[p % len(block_pool)])
    for idx in range(n_patches, len(block_pool)):
        patch_blocks[idx % n_patches].append(block_pool[idx])

    all_assigned = [b for pb in patch_blocks for b in pb]
    for p in range(n_patches):
        n_extra = rng.integers(0, max_blocks_per_patch)
        for _ in range(n_extra):
            if rng.random() < feature_redundancy and all_assigned:
                donor_block = all_assigned[rng.integers(len(all_assigned))]
                patch_blocks[p].append(donor_block)
            else:
                start = rng.integers(0, max(1, N - MIN_PATCH_COLS))
                width = rng.integers(MIN_PATCH_COLS,
                                     max(MIN_PATCH_COLS + 1, N // n_patches + 10))
                end = min(N, start + width)
                patch_blocks[p].append((start, end))

    patch_features = []
    for p in range(n_patches):
        cols = np.unique(np.concatenate(
            [np.arange(s, e) for s, e in patch_blocks[p]]))
        while len(cols) < MIN_PATCH_COLS:
            start = rng.integers(0, max(1, N - MIN_PATCH_COLS))
            extra = np.arange(start, min(N, start + MIN_PATCH_COLS))
            cols = np.unique(np.concatenate([cols, extra]))
        patch_features.append(cols)

    # Sample assignment
    raw_sizes = rng.uniform(0.5, 1.5, size=n_patches)
    core_sizes = np.round(raw_sizes / raw_sizes.sum() * M).astype(int)
    core_sizes = np.maximum(core_sizes, MIN_PATCH_ROWS)
    core_sizes = np.round(core_sizes / core_sizes.sum() * M).astype(int)
    core_sizes[-1] = M - core_sizes[:-1].sum()
    breakpoints = np.concatenate([[0], np.cumsum(core_sizes)])

    patches = []
    for i in range(n_patches):
        core_start = breakpoints[i]
        core_end = breakpoints[i + 1]
        core_size = core_end - core_start
        row_start, row_end = core_start, core_end

        if i > 0:
            ov = overlap_frac * rng.uniform(0.5, 1.5)
            ov_size = int(core_size * ov)
            row_start = max(0, core_start - ov_size)
        if i < n_patches - 1:
            ov = overlap_frac * rng.uniform(0.5, 1.5)
            ov_size = int(core_size * ov)
            row_end = min(M, core_end + ov_size)
        if row_end - row_start < MIN_PATCH_ROWS:
            row_end = min(M, row_start + MIN_PATCH_ROWS)

        patches.append({
            'row_idx': np.arange(row_start, row_end),
            'col_idx': patch_features[i],
        })
    return patches


# ── Greedy patch ordering (from generate_quilt.ipynb cell-12) ─────────────

def greedy_patch_ordering(patches_data):
    """Order patches by greedy max sample-overlap with accumulated set."""
    n = len(patches_data)
    if n == 1:
        return [0]

    row_sets = [set(p['row_idx'].tolist()) for p in patches_data]

    best_score, best_pair = -1, (0, 1)
    for i in range(n):
        for j in range(n):
            if i != j:
                s = len(row_sets[i] & row_sets[j])
                if s > best_score:
                    best_score, best_pair = s, (i, j)

    ordering = [best_pair[0], best_pair[1]]
    used = set(ordering)
    acc_rows = row_sets[best_pair[0]] | row_sets[best_pair[1]]

    for _ in range(2, n):
        best_next, best_s = -1, -1
        for j in range(n):
            if j not in used:
                s = len(row_sets[j] & acc_rows)
                if s > best_s:
                    best_s, best_next = s, j
        ordering.append(best_next)
        used.add(best_next)
        acc_rows |= row_sets[best_next]
    return ordering


# ── Sequential quilting (from generate_quilt.ipynb cell-12) ───────────────

def sequential_quilting(patches_data, X_full, r, K):
    """Algorithm 1: sequential chain quilting. Calls ordering internally."""
    ordering = greedy_patch_ordering(patches_data)
    M_total = X_full.shape[0]
    U_tilde = np.zeros((M_total, r))
    covered_rows = set()

    m0 = ordering[0]
    row_idx = patches_data[m0]['row_idx']
    col_idx = patches_data[m0]['col_idx']
    X_m = X_full[np.ix_(row_idx, col_idx)]
    U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
    U_tilde[row_idx, :] = U[:, :r]
    covered_rows.update(row_idx.tolist())

    for step in range(1, len(ordering)):
        m = ordering[step]
        row_idx = patches_data[m]['row_idx']
        col_idx = patches_data[m]['col_idx']
        X_m = X_full[np.ix_(row_idx, col_idx)]
        U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
        U_r = U[:, :r]

        current_rows = set(row_idx.tolist())
        overlap_global = sorted(current_rows & covered_rows)

        if len(overlap_global) < r:
            U_tilde[row_idx, :] = U_r
            covered_rows.update(current_rows)
            continue

        row_list = row_idx.tolist()
        g2l = {g: l for l, g in enumerate(row_list)}
        local_overlap = [g2l[g] for g in overlap_global]

        G, _, _, _ = np.linalg.lstsq(
            U_r[local_overlap, :], U_tilde[overlap_global, :], rcond=None)

        new_global = sorted(current_rows - covered_rows)
        if new_global:
            new_local = [g2l[g] for g in new_global]
            U_tilde[new_global, :] = U_r[new_local, :] @ G

        covered_rows.update(current_rows)

    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(U_tilde)
    return labels, U_tilde


# ── Full data baseline (from generate_quilt.ipynb cell-12) ────────────────

def full_data_baseline(X_data, labels_true, K, r):
    """PCA + k-means on fully observed data. Returns (ARI, embedding)."""
    pca = PCA(n_components=r)
    X_pca = pca.fit_transform(X_data)
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    pred = km.fit_predict(X_pca)
    return adjusted_rand_score(labels_true, pred), X_pca


# ── Hierarchical quilting — new functions ─────────────────────────────────

def _greedy_pairing(nodes):
    """Pair nodes by maximum sample overlap (greedy)."""
    n = len(nodes)
    if n == 1:
        return [], 0

    row_sets = [set(nd['row_list'].tolist()) for nd in nodes]
    used = set()
    pairs = []

    scored = []
    for i in range(n):
        for j in range(i + 1, n):
            scored.append((len(row_sets[i] & row_sets[j]), i, j))
    scored.sort(reverse=True)

    for _, i, j in scored:
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.add(i)
        used.add(j)
        if len(used) >= n - 1:
            break

    unpaired = None
    for k in range(n):
        if k not in used:
            unpaired = k
            break

    return pairs, unpaired


def _merge_nodes(node_a, node_b, r):
    """Merge two nodes by aligning node_b onto node_a via shared samples."""
    rows_a = set(node_a['row_list'].tolist())
    rows_b = set(node_b['row_list'].tolist())
    overlap = sorted(rows_a & rows_b)

    g2l_a = {g: l for l, g in enumerate(node_a['row_list'].tolist())}
    g2l_b = {g: l for l, g in enumerate(node_b['row_list'].tolist())}

    if len(overlap) >= r:
        local_ov_a = [g2l_a[g] for g in overlap]
        local_ov_b = [g2l_b[g] for g in overlap]

        U_a_ov = node_a['U_dense'][local_ov_a, :]
        U_b_ov = node_b['U_dense'][local_ov_b, :]

        G, _, _, _ = np.linalg.lstsq(U_b_ov, U_a_ov, rcond=None)
        U_b_aligned = node_b['U_dense'] @ G
    else:
        U_b_aligned = node_b['U_dense']

    new_in_b = sorted(rows_b - rows_a)
    union_rows = list(node_a['row_list']) + new_in_b
    union_rows_arr = np.array(union_rows, dtype=np.int64)

    n_a = len(node_a['row_list'])
    n_new = len(new_in_b)
    U_merged = np.empty((n_a + n_new, r), dtype=np.float64)
    U_merged[:n_a, :] = node_a['U_dense']

    if n_new > 0:
        new_local_b = [g2l_b[g] for g in new_in_b]
        U_merged[n_a:, :] = U_b_aligned[new_local_b, :]

    return {'U_dense': U_merged, 'row_list': union_rows_arr}


def hierarchical_quilting(patches_data, X_full, r, K):
    """Hierarchical (bottom-up pairwise) cluster quilting."""
    N_total = X_full.shape[0]

    nodes = []
    for p in patches_data:
        row_idx = p['row_idx']
        col_idx = p['col_idx']
        X_m = X_full[np.ix_(row_idx, col_idx)]
        U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
        nodes.append({
            'U_dense': U[:, :r].copy(),
            'row_list': np.array(row_idx, dtype=np.int64),
        })

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes)
        next_level = []

        for i, j in pairs:
            if len(nodes[i]['row_list']) >= len(nodes[j]['row_list']):
                merged = _merge_nodes(nodes[i], nodes[j], r)
            else:
                merged = _merge_nodes(nodes[j], nodes[i], r)
            next_level.append(merged)

        if unpaired is not None:
            next_level.append(nodes[unpaired])

        nodes = next_level

    final = nodes[0]
    U_tilde = np.zeros((N_total, r))
    U_tilde[final['row_list'], :] = final['U_dense']

    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(U_tilde)
    return labels, U_tilde


# ── Comparison sweep (with replicates) ───────────────────────────────────

def run_comparison_sweep(X_data, labels_true, shape, K, r,
                         n_patches_list, overlap_list, dataset_name,
                         n_replicates=3, seeds=None):
    """Run both methods across the full (n_patches x overlap) grid with replicates.

    Returns
    -------
    ari_seq : ndarray (n_rows, n_cols, n_replicates)
    ari_hier : ndarray (n_rows, n_cols, n_replicates)
    ari_baseline : float
    all_n : list — row labels (0 = baseline, then n_patches values)
    all_ov : list — column labels
    emb_seq : dict (i, j) -> ndarray — embeddings from seed 0
    emb_hier : dict (i, j) -> ndarray — embeddings from seed 0
    """
    if seeds is None:
        seeds = list(range(n_replicates))

    all_n = [0] + n_patches_list
    all_ov = overlap_list if 0.0 in overlap_list else [0.0] + overlap_list
    n_rows = len(all_n)
    n_cols = len(all_ov)

    ari_seq = np.zeros((n_rows, n_cols, n_replicates))
    ari_hier = np.zeros((n_rows, n_cols, n_replicates))
    emb_seq = {}
    emb_hier = {}

    # Baseline (row 0) — deterministic, same for all replicates
    baseline_ari, baseline_emb = full_data_baseline(X_data, labels_true, K, r)
    ari_seq[0, :, :] = baseline_ari
    ari_hier[0, :, :] = baseline_ari
    for j in range(n_cols):
        emb_seq[(0, j)] = baseline_emb
        emb_hier[(0, j)] = baseline_emb
    print(f'  [baseline]  ARI={baseline_ari:.4f}  (PCA r={r} + k-means)')

    total_configs = len(n_patches_list) * len(all_ov) * n_replicates
    run_count = 0
    t_start = time.time()

    for i, n_p in enumerate(n_patches_list, start=1):
        for j, ov in enumerate(all_ov):
            for rep, seed in enumerate(seeds):
                p_list = generate_patches(
                    shape=shape, n_patches=n_p, overlap_frac=ov,
                    feature_redundancy=0.15, rng=np.random.default_rng(seed))

                pred_seq, U_seq = sequential_quilting(p_list, X_data, r=r, K=K)
                a_seq = adjusted_rand_score(labels_true, pred_seq)

                pred_hier, U_hier = hierarchical_quilting(p_list, X_data, r=r, K=K)
                a_hier = adjusted_rand_score(labels_true, pred_hier)

                ari_seq[i, j, rep] = a_seq
                ari_hier[i, j, rep] = a_hier

                # Store embeddings for seed 0 only (for 3D plots)
                if rep == 0:
                    emb_seq[(i, j)] = U_seq
                    emb_hier[(i, j)] = U_hier

                run_count += 1
                elapsed = time.time() - t_start
                eta = elapsed / run_count * (total_configs - run_count)

            # Print summary per (n_p, ov) after all replicates
            tag = f'{n_p}p_{int(ov * 100):02d}ov'
            s_mean = ari_seq[i, j, :].mean()
            h_mean = ari_hier[i, j, :].mean()
            print(f'  [{run_count:>4d}/{total_configs}] {tag:>10s}  '
                  f'seq={s_mean:.4f}  hier={h_mean:.4f}  '
                  f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

    return ari_seq, ari_hier, baseline_ari, all_n, all_ov, emb_seq, emb_hier


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_side_by_side_heatmaps(ari_seq_mean, ari_hier_mean,
                               ari_seq_std, ari_hier_std,
                               all_n, all_ov, title, save_path):
    """Side-by-side ARI heatmaps: sequential vs hierarchical (mean +/- std)."""
    vmax = max(0.3, max(ari_seq_mean.max(), ari_hier_mean.max()) * 1.1)

    fig = plt.figure(figsize=(23, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)

    for idx, (ari_mean, ari_std, subtitle) in enumerate(zip(
            [ari_seq_mean, ari_hier_mean],
            [ari_seq_std, ari_hier_std],
            ['Sequential', 'Hierarchical'])):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(ari_mean, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(all_ov)))
        ax.set_xticklabels([f'{ov:.0%}' for ov in all_ov])
        ax.set_yticks(range(len(all_n)))
        ax.set_yticklabels([str(n) if n > 0 else 'Full data' for n in all_n])
        ax.set_xlabel('Sample overlap fraction')
        ax.set_ylabel('Number of patches')
        ax.set_title(subtitle)

        for i in range(len(all_n)):
            for j in range(len(all_ov)):
                val = ari_mean[i, j]
                sd = ari_std[i, j]
                color = 'white' if val > vmax * 0.55 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)
                if sd > 0:
                    ax.text(j, i + 0.3, f'\u00b1{sd:.3f}', ha='center', va='center',
                            fontsize=6, color=color, alpha=0.7)

    cbar_ax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im, cax=cbar_ax, label='Adjusted Rand Index')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_ari_line_comparison(ari_seq_raw, ari_hier_raw, baseline_raw,
                             ari_seq_sim, ari_hier_sim, baseline_sim,
                             n_patches_list, overlap_list, save_path):
    """Line plots: ARI vs n_patches with SEM error bands.

    2 rows (raw, simulated) x 5 cols (one per overlap level).
    ari arrays are 3D: (n_rows, n_cols, n_replicates).
    """
    all_ov = overlap_list if 0.0 in overlap_list else [0.0] + overlap_list
    ov_to_j = {ov: j for j, ov in enumerate(all_ov)}
    n_rep = ari_seq_raw.shape[2]

    fig, axes = plt.subplots(2, len(all_ov), figsize=(4.5 * len(all_ov), 9),
                             sharey='row')

    for row_idx, (ari_s, ari_h, bl, ds_name) in enumerate([
            (ari_seq_raw, ari_hier_raw, baseline_raw, 'Raw Data'),
            (ari_seq_sim, ari_hier_sim, baseline_sim, 'Simulated')]):

        for col_idx, ov in enumerate(all_ov):
            ax = axes[row_idx, col_idx]
            j = ov_to_j[ov]

            # Extract mean and SEM across replicates (skip row 0 = baseline)
            seq_all = np.array([ari_s[i, j, :] for i in range(1, len(n_patches_list) + 1)])
            hier_all = np.array([ari_h[i, j, :] for i in range(1, len(n_patches_list) + 1)])

            seq_mean = seq_all.mean(axis=1)
            seq_sem = seq_all.std(axis=1) / np.sqrt(n_rep)
            hier_mean = hier_all.mean(axis=1)
            hier_sem = hier_all.std(axis=1) / np.sqrt(n_rep)

            x = np.array(n_patches_list)

            ax.plot(x, seq_mean, 'o-', color='#e41a1c',
                    label='Sequential', linewidth=2, markersize=4)
            ax.fill_between(x, seq_mean - seq_sem, seq_mean + seq_sem,
                            color='#e41a1c', alpha=0.2)

            ax.plot(x, hier_mean, 's--', color='#377eb8',
                    label='Hierarchical', linewidth=2, markersize=4)
            ax.fill_between(x, hier_mean - hier_sem, hier_mean + hier_sem,
                            color='#377eb8', alpha=0.2)

            ax.axhline(bl, color='#999999', linestyle=':', linewidth=1.5,
                       label='Full-data baseline')

            ax.set_xlabel('Number of patches')
            ax.set_title(f'{ds_name} \u2014 {ov:.0%} overlap')
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) if v % 3 == 0 else '' for v in x],
                               fontsize=7)
            if col_idx == 0:
                ax.set_ylabel('ARI')
            if row_idx == 0 and col_idx == len(all_ov) - 1:
                ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_3d_embedding_grid(embeddings, ari_mean, all_n, all_ov,
                           labels_true, n_patches_subset, method_name,
                           title, save_path, n_subsample=500):
    """Plot 3D scatter grid of quilted embeddings (seed 0), colored by true labels.

    Shows a subsampled set of n_patches values x all overlap levels.
    """
    rng_plot = np.random.default_rng(42)
    unique_labels = np.unique(labels_true)
    K = len(unique_labels)
    colors_map = {lab: c for lab, c in zip(
        unique_labels,
        ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3'][:K]
    )}

    # Subsample indices (balanced across classes)
    sub_idx = []
    for lab in unique_labels:
        idx = np.where(labels_true == lab)[0]
        chosen = rng_plot.choice(idx, size=min(n_subsample, len(idx)), replace=False)
        sub_idx.append(chosen)
    sub_idx = np.concatenate(sub_idx)
    labels_sub = labels_true[sub_idx]

    # Map n_patches_subset values to row indices in all_n
    n_to_row = {n: i for i, n in enumerate(all_n)}
    row_subset = [0] + [n_to_row[n] for n in n_patches_subset if n in n_to_row]
    row_labels = ['Full data'] + [f'{n}p' for n in n_patches_subset if n in n_to_row]

    nr = len(row_subset)
    nc = len(all_ov)
    fig = plt.figure(figsize=(5 * nc, 4.5 * nr))

    for ri, row_i in enumerate(row_subset):
        for ci in range(nc):
            ax = fig.add_subplot(nr, nc, ri * nc + ci + 1, projection='3d')
            key = (row_i, ci)
            if key not in embeddings:
                ax.set_title('N/A', fontsize=9)
                continue

            emb = embeddings[key][sub_idx]
            for lab in unique_labels:
                mask = labels_sub == lab
                ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                           s=3, alpha=0.3, color=colors_map[lab],
                           label=str(lab))

            ov_label = f'{all_ov[ci]:.0%} ov'
            ari_val = ari_mean[row_i, ci]
            ax.set_title(f'{row_labels[ri]}, {ov_label}\nARI={ari_val:.3f}',
                         fontsize=9)
            ax.tick_params(labelsize=6)

            if ri == 0 and ci == nc - 1:
                ax.legend(markerscale=4, fontsize=7, loc='upper right')

    fig.suptitle(f'{title} ({method_name})', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Statistical tests ────────────────────────────────────────────────────

def run_statistical_tests(ari_seq, ari_hier, all_n, all_ov, n_patches_list):
    """Paired t-tests: sequential vs hierarchical at each grid point.

    Parameters
    ----------
    ari_seq, ari_hier : ndarray (n_rows, n_cols, n_replicates)

    Returns
    -------
    pvalues : ndarray (n_rows, n_cols) — NaN for baseline row
    significant : ndarray (n_rows, n_cols) of bool — Bonferroni-corrected
    diff_mean : ndarray (n_rows, n_cols) — mean(hier) - mean(seq)
    """
    n_rows, n_cols, n_rep = ari_seq.shape
    pvalues = np.full((n_rows, n_cols), np.nan)
    diff_mean = np.zeros((n_rows, n_cols))

    n_tests = len(n_patches_list) * n_cols  # skip baseline row
    alpha_corrected = 0.05 / n_tests  # Bonferroni

    for i in range(1, n_rows):  # skip baseline
        for j in range(n_cols):
            seq_vals = ari_seq[i, j, :]
            hier_vals = ari_hier[i, j, :]
            diff_mean[i, j] = hier_vals.mean() - seq_vals.mean()

            # Need variance to run t-test; skip if all values identical
            if np.allclose(seq_vals, hier_vals):
                pvalues[i, j] = 1.0
            else:
                _, p = ttest_rel(hier_vals, seq_vals)
                pvalues[i, j] = p

    significant = pvalues < alpha_corrected

    # Print summary
    print(f'\n  Paired t-test summary (Bonferroni alpha={alpha_corrected:.6f}, '
          f'n={n_rep} replicates):')
    print(f'  {"Config":>12s}  {"diff(H-S)":>10s}  {"p-value":>10s}  {"sig":>4s}')
    print(f'  {"-"*12}  {"-"*10}  {"-"*10}  {"-"*4}')
    for i in range(1, n_rows):
        for j in range(n_cols):
            tag = f'{all_n[i]}p_{int(all_ov[j]*100):02d}ov'
            d = diff_mean[i, j]
            p = pvalues[i, j]
            s = '*' if significant[i, j] else ''
            if abs(d) > 0.01:
                print(f'  {tag:>12s}  {d:>+10.4f}  {p:>10.4f}  {s:>4s}')

    n_sig = np.nansum(significant)
    print(f'\n  {int(n_sig)}/{n_tests} tests significant after Bonferroni correction')
    print(f'  (Note: n={n_rep} replicates provides limited statistical power)')

    return pvalues, significant, diff_mean


def plot_effect_size_heatmap(diff_mean, pvalues, significant,
                             all_n, all_ov, title, save_path):
    """Heatmap of mean(hier) - mean(seq) with significance markers."""
    # Skip baseline row for the plot
    diff_plot = diff_mean[1:, :]
    sig_plot = significant[1:, :]
    row_labels = [str(n) for n in all_n[1:]]
    col_labels = [f'{ov:.0%}' for ov in all_ov]

    vabs = max(0.05, np.abs(diff_plot).max() * 1.1)

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(diff_plot, cmap='RdBu', aspect='auto',
                   vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Sample overlap fraction')
    ax.set_ylabel('Number of patches')

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = diff_plot[i, j]
            color = 'white' if abs(val) > vabs * 0.55 else 'black'
            text = f'{val:+.3f}'
            if sig_plot[i, j]:
                text += ' *'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=7, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, label='ARI difference (Hierarchical - Sequential)',
                 shrink=0.6)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    Path('plots').mkdir(exist_ok=True)

    n_patches_list = list(range(1, 16))  # 1 through 15
    overlap_list = [0.0, 0.10, 0.20, 0.40, 0.80]
    n_replicates = 5
    seeds = list(range(n_replicates))
    n_patches_subset = [1, 3, 5, 8, 12, 15]  # for 3D embedding plots

    # ── Load raw data ─────────────────────────────────────────────────
    print('Loading raw data...')
    df = pd.read_csv('trialdf_24sessions.csv', index_col=0)
    meta_cols = ['valence', 'airstart', 'sucstart', 'ms_id',
                 'condition', 'inj_site', 'ms_n']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df.loc[~df['valence'].str.contains('CS', na=False)]
    thresh = 0.05 * len(df)
    feature_cols = [c for c in feature_cols if df[c].isna().sum() <= thresh]
    df = df[feature_cols + meta_cols].dropna()

    valence = df['valence'].values
    X_raw = StandardScaler().fit_transform(df[feature_cols].values)
    K_raw = len(np.unique(valence))
    r_raw = K_raw
    print(f'  Raw: {X_raw.shape[0]} x {X_raw.shape[1]}, K={K_raw}')

    # ── Load simulated data ───────────────────────────────────────────
    print('Loading simulated data...')
    sim = np.load('simulated.npz')
    X_sim = sim['data']
    labels_sim = sim['labels']
    K_sim = len(np.unique(labels_sim))
    r_sim = K_sim
    print(f'  Sim: {X_sim.shape[0]} x {X_sim.shape[1]}, K={K_sim}')

    # ── Raw sweep ─────────────────────────────────────────────────────
    print(f'\n=== Raw Data Sweep ({len(n_patches_list)} x {len(overlap_list)} '
          f'x {n_replicates} replicates) ===')
    (ari_seq_raw, ari_hier_raw, bl_raw, all_n, all_ov,
     emb_seq_raw, emb_hier_raw) = run_comparison_sweep(
        X_raw, valence, X_raw.shape, K_raw, r_raw,
        n_patches_list, overlap_list, 'raw',
        n_replicates=n_replicates, seeds=seeds)

    # Heatmaps
    plot_side_by_side_heatmaps(
        ari_seq_raw.mean(axis=2), ari_hier_raw.mean(axis=2),
        ari_seq_raw.std(axis=2), ari_hier_raw.std(axis=2),
        all_n, all_ov,
        f'Sequential vs Hierarchical ARI — Raw Data '
        f'({X_raw.shape[0]} x {X_raw.shape[1]}, n={n_replicates})',
        'plots/heatmap_comparison_raw.png')

    # Stats
    print('\n--- Raw Data Statistical Tests ---')
    pvals_raw, sig_raw, diff_raw = run_statistical_tests(
        ari_seq_raw, ari_hier_raw, all_n, all_ov, n_patches_list)

    plot_effect_size_heatmap(
        diff_raw, pvals_raw, sig_raw, all_n, all_ov,
        'Hierarchical \u2212 Sequential ARI Difference \u2014 Raw Data',
        'plots/effect_size_raw.png')

    # 3D embeddings (seed 0)
    plot_3d_embedding_grid(
        emb_seq_raw, ari_seq_raw.mean(axis=2), all_n, all_ov,
        valence, n_patches_subset, 'Sequential',
        f'Quilted Embeddings \u2014 Raw Data ({X_raw.shape[0]} x {X_raw.shape[1]})',
        'plots/embeddings_3d_seq_raw.png')
    plot_3d_embedding_grid(
        emb_hier_raw, ari_hier_raw.mean(axis=2), all_n, all_ov,
        valence, n_patches_subset, 'Hierarchical',
        f'Quilted Embeddings \u2014 Raw Data ({X_raw.shape[0]} x {X_raw.shape[1]})',
        'plots/embeddings_3d_hier_raw.png')

    # ── Simulated sweep ───────────────────────────────────────────────
    print(f'\n=== Simulated Data Sweep ({len(n_patches_list)} x {len(overlap_list)} '
          f'x {n_replicates} replicates) ===')
    (ari_seq_sim, ari_hier_sim, bl_sim, all_n_s, all_ov_s,
     emb_seq_sim, emb_hier_sim) = run_comparison_sweep(
        X_sim, labels_sim, X_sim.shape, K_sim, r_sim,
        n_patches_list, overlap_list, 'simulated',
        n_replicates=n_replicates, seeds=seeds)

    # Heatmaps
    plot_side_by_side_heatmaps(
        ari_seq_sim.mean(axis=2), ari_hier_sim.mean(axis=2),
        ari_seq_sim.std(axis=2), ari_hier_sim.std(axis=2),
        all_n_s, all_ov_s,
        f'Sequential vs Hierarchical ARI — Simulated '
        f'({X_sim.shape[0]} x {X_sim.shape[1]}, n={n_replicates})',
        'plots/heatmap_comparison_simulated.png')

    # Stats
    print('\n--- Simulated Data Statistical Tests ---')
    pvals_sim, sig_sim, diff_sim = run_statistical_tests(
        ari_seq_sim, ari_hier_sim, all_n_s, all_ov_s, n_patches_list)

    plot_effect_size_heatmap(
        diff_sim, pvals_sim, sig_sim, all_n_s, all_ov_s,
        'Hierarchical \u2212 Sequential ARI Difference \u2014 Simulated',
        'plots/effect_size_sim.png')

    # 3D embeddings (seed 0)
    plot_3d_embedding_grid(
        emb_seq_sim, ari_seq_sim.mean(axis=2), all_n_s, all_ov_s,
        labels_sim, n_patches_subset, 'Sequential',
        f'Quilted Embeddings \u2014 Simulated ({X_sim.shape[0]} x {X_sim.shape[1]})',
        'plots/embeddings_3d_seq_sim.png')
    plot_3d_embedding_grid(
        emb_hier_sim, ari_hier_sim.mean(axis=2), all_n_s, all_ov_s,
        labels_sim, n_patches_subset, 'Hierarchical',
        f'Quilted Embeddings \u2014 Simulated ({X_sim.shape[0]} x {X_sim.shape[1]})',
        'plots/embeddings_3d_hier_sim.png')

    # ── Line comparison ───────────────────────────────────────────────
    plot_ari_line_comparison(
        ari_seq_raw, ari_hier_raw, bl_raw,
        ari_seq_sim, ari_hier_sim, bl_sim,
        n_patches_list, overlap_list,
        'plots/ari_line_comparison.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
