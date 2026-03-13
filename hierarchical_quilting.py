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

def generate_patches(shape, n_patches, overlap_frac=0.25, rng=None):
    """Generate patchwork patches with scattered row blocks.

    Each patch gets one contiguous column range but *scattered*
    (non-contiguous) row blocks drawn from across the full sample space.
    This matches the paper's architecture (Zheng, Chang & Allen 2024,
    Fig. 1) and creates a dense sample-overlap graph where each patch
    shares rows with many other patches — not just its immediate
    neighbors.

    Row assignment:
        The sample axis is divided into ``3 * n_patches`` blocks.  Each
        block is assigned a primary patch via shuffled round-robin (so
        primary blocks are scattered, not adjacent).  Each block is then
        shared with ``n_extra`` additional patches chosen at random,
        where ``n_extra`` scales with ``overlap_frac``.

    Column assignment:
        Identical to before — evenly spaced centers, shared width,
        iteratively shrunk until no row has all N features observed.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    M, N = shape

    # ── Single patch: cover most of the matrix ─────────────────────────
    if n_patches == 1:
        col_width = max(MIN_PATCH_COLS, int(N * 0.80))
        center = N / 2
        cs = max(0, int(round(center - col_width / 2)))
        ce = min(N, cs + col_width)
        return [{'row_idx': np.arange(M), 'col_idx': np.arange(cs, ce)}]

    # ── Row assignment: scattered blocks ───────────────────────────────
    n_blocks = n_patches * 3
    block_bounds = np.linspace(0, M, n_blocks + 1, dtype=int)
    block_sizes = np.diff(block_bounds)

    # Primary assignment: round-robin, shuffled so each patch's primary
    # blocks are scattered across the sample space.
    primary = np.arange(n_blocks) % n_patches
    rng.shuffle(primary)

    # (n_patches, n_blocks) boolean assignment matrix
    assigned = np.zeros((n_patches, n_blocks), dtype=bool)
    assigned[primary, np.arange(n_blocks)] = True

    # Overlap: each block is shared with additional patches
    # probabilistically.  Each non-primary patch includes the block with
    # probability ``overlap_frac``, capped so no block belongs to more
    # than ``max_per_block`` patches (keeps columns wide enough).
    max_per_block = max(2, N // MIN_PATCH_COLS - 1)

    if overlap_frac > 0:
        # Draw all random values at once
        rand_vals = rng.random((n_blocks, n_patches))
        for b in range(n_blocks):
            budget = max_per_block - 1
            order = rng.permutation(n_patches)
            for p in order:
                if p == primary[b]:
                    continue
                if budget <= 0:
                    break
                if rand_vals[b, p] < overlap_frac:
                    assigned[p, b] = True
                    budget -= 1

    # Ensure minimum row count per patch.
    for p in range(n_patches):
        total = block_sizes[assigned[p]].sum()
        while total < MIN_PATCH_ROWS:
            avail = np.where(~assigned[p])[0]
            if len(avail) == 0:
                break
            b = rng.choice(avail)
            assigned[p, b] = True
            total += block_sizes[b]

    # Convert assignment matrix → sorted row index arrays.
    row_indices = []
    for p in range(n_patches):
        blocks = np.where(assigned[p])[0]
        rows = np.concatenate([np.arange(block_bounds[b], block_bounds[b + 1])
                               for b in blocks])
        row_indices.append(rows)

    # ── Column assignment: centers spread evenly across [0, N) ─────────
    # Compute actual max patches per row from the assignment matrix.
    # Expand block counts to per-row counts via repeat.
    patches_per_block = assigned.sum(axis=0)        # (n_blocks,)
    row_membership = np.repeat(patches_per_block, block_sizes)
    k_max = max(1, int(row_membership.max()))

    stride = N / n_patches
    safe_w = max(1, int(N / k_max) - 1)
    col_width = max(1, min(int(N * 0.80), safe_w, N - 1))
    col_width = max(col_width, min(MIN_PATCH_COLS, safe_w))

    while col_width >= 1:
        col_ranges = []
        for p in range(n_patches):
            center = stride / 2 + p * stride
            cs = max(0, int(round(center - col_width / 2)))
            ce = min(N, cs + col_width)
            if ce - cs < col_width:
                cs = max(0, ce - col_width)
            col_ranges.append((cs, ce))

        # Coverage check: no row may have all N features observed.
        observed = np.zeros((M, N), dtype=bool)
        for p in range(n_patches):
            cs, ce = col_ranges[p]
            observed[np.ix_(row_indices[p], np.arange(cs, ce))] = True
        if not observed.all(axis=1).any():
            break
        col_width -= 1

    # ── Assemble patches ──────────────────────────────────────────────
    patches = []
    for p in range(n_patches):
        cs, ce = col_ranges[p]
        patches.append({
            'row_idx': row_indices[p],
            'col_idx': np.arange(cs, ce),
        })

    return patches


# ── Greedy patch ordering (from generate_quilt.ipynb cell-12) ─────────────

def _overlap_matrix(patches_data, M):
    """Pairwise sample-overlap counts via boolean membership matrix."""
    n = len(patches_data)
    # (n, M) boolean membership matrix
    mem = np.zeros((n, M), dtype=np.float32)
    for i, p in enumerate(patches_data):
        mem[i, p['row_idx']] = 1.0
    # overlap[i, j] = number of shared rows
    return (mem @ mem.T).astype(np.int64), mem


def greedy_patch_ordering(patches_data, M=None):
    """Order patches by greedy max sample-overlap with accumulated set."""
    n = len(patches_data)
    if n == 1:
        return [0]

    if M is None:
        M = max(p['row_idx'].max() for p in patches_data) + 1
    overlap_mat, mem = _overlap_matrix(patches_data, M)

    # Best starting pair
    np.fill_diagonal(overlap_mat, -1)
    flat_idx = np.argmax(overlap_mat)
    i0, j0 = divmod(flat_idx, n)
    np.fill_diagonal(overlap_mat, 0)

    ordering = [int(i0), int(j0)]
    used = np.zeros(n, dtype=bool)
    used[i0] = used[j0] = True
    # Accumulated row membership (float32 for matmul compat)
    acc = mem[i0] + mem[j0]
    acc = np.minimum(acc, 1.0)

    for _ in range(2, n):
        # Overlap of each unused patch with accumulated set
        scores = mem @ acc          # (n,) dot products
        scores[used] = -1
        best = int(np.argmax(scores))
        ordering.append(best)
        used[best] = True
        acc = np.minimum(acc + mem[best], 1.0)

    return ordering


# ── Sequential quilting (from generate_quilt.ipynb cell-12) ───────────────

def sequential_quilting(patches_data, X_full, r, K):
    """Algorithm 1: sequential chain quilting. Calls ordering internally."""
    M_total = X_full.shape[0]
    ordering = greedy_patch_ordering(patches_data, M=M_total)
    U_tilde = np.zeros((M_total, r))
    covered = np.zeros(M_total, dtype=bool)

    m0 = ordering[0]
    row_idx = patches_data[m0]['row_idx']
    col_idx = patches_data[m0]['col_idx']
    X_m = X_full[np.ix_(row_idx, col_idx)]
    U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
    U_tilde[row_idx, :] = U[:, :r]
    covered[row_idx] = True

    for step in range(1, len(ordering)):
        m = ordering[step]
        row_idx = patches_data[m]['row_idx']   # sorted
        col_idx = patches_data[m]['col_idx']
        X_m = X_full[np.ix_(row_idx, col_idx)]
        U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
        U_r = U[:, :r]

        # Boolean masks for overlap / new rows (vectorised)
        is_covered = covered[row_idx]
        overlap_global = row_idx[is_covered]
        new_global = row_idx[~is_covered]

        if len(overlap_global) < r:
            U_tilde[row_idx, :] = U_r
            covered[row_idx] = True
            continue

        # Map global overlap rows → local indices via searchsorted
        overlap_local = np.searchsorted(row_idx, overlap_global)

        G, _, _, _ = np.linalg.lstsq(
            U_r[overlap_local, :], U_tilde[overlap_global, :], rcond=None)

        if len(new_global) > 0:
            new_local = np.searchsorted(row_idx, new_global)
            U_tilde[new_global, :] = U_r[new_local, :] @ G

        covered[row_idx] = True

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

def _greedy_pairing(nodes, N_total):
    """Pair nodes by maximum sample overlap (greedy), vectorised."""
    n = len(nodes)
    if n == 1:
        return [], 0

    # Build boolean membership matrix and compute pairwise overlaps
    mem = np.zeros((n, N_total), dtype=np.float32)
    for i, nd in enumerate(nodes):
        mem[i, nd['row_list']] = 1.0
    overlap_mat = (mem @ mem.T).astype(np.int64)
    np.fill_diagonal(overlap_mat, -1)

    # Greedy matching: pick best remaining pair repeatedly
    used = np.zeros(n, dtype=bool)
    pairs = []
    # Flatten + argsort once (descending)
    flat_order = np.argsort(overlap_mat.ravel())[::-1]

    for flat_idx in flat_order:
        i, j = divmod(int(flat_idx), n)
        if i >= j or used[i] or used[j]:
            continue
        pairs.append((i, j))
        used[i] = used[j] = True
        if used.sum() >= n - 1:
            break

    unpaired = None
    remaining = np.where(~used)[0]
    if len(remaining) == 1:
        unpaired = int(remaining[0])

    return pairs, unpaired


def _merge_nodes(node_a, node_b, r):
    """Merge two nodes by aligning node_b onto node_a via shared samples."""
    rows_a = node_a['row_list']        # sorted int64 array
    rows_b = node_b['row_list']        # sorted int64 array

    overlap = np.intersect1d(rows_a, rows_b, assume_unique=True)

    if len(overlap) >= r:
        local_ov_a = np.searchsorted(rows_a, overlap)
        local_ov_b = np.searchsorted(rows_b, overlap)

        G, _, _, _ = np.linalg.lstsq(
            node_b['U_dense'][local_ov_b, :],
            node_a['U_dense'][local_ov_a, :], rcond=None)
        U_b_aligned = node_b['U_dense'] @ G
    else:
        U_b_aligned = node_b['U_dense']

    new_in_b = np.setdiff1d(rows_b, rows_a, assume_unique=True)

    n_a = len(rows_a)
    n_new = len(new_in_b)
    U_cat = np.empty((n_a + n_new, r), dtype=np.float64)
    U_cat[:n_a, :] = node_a['U_dense']

    if n_new > 0:
        new_local_b = np.searchsorted(rows_b, new_in_b)
        U_cat[n_a:, :] = U_b_aligned[new_local_b, :]

    # Keep row_list sorted so searchsorted works in subsequent merges.
    union_rows = np.concatenate([rows_a, new_in_b])
    order = np.argsort(union_rows)
    return {'U_dense': U_cat[order, :], 'row_list': union_rows[order]}


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
            'row_list': np.asarray(row_idx, dtype=np.int64),
        })

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes, N_total)
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
    all_ov = list(overlap_list)
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
                    rng=np.random.default_rng(seed))

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
    all_ov = list(overlap_list)
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
            seq_all = ari_s[1:, j, :]
            hier_all = ari_h[1:, j, :]

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
    Path('plots/new_scattered').mkdir(parents=True, exist_ok=True)

    n_patches_list = list(range(1, 16))  # 1 through 15
    overlap_list = [0.10, 0.20, 0.40, 0.80]
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
        'plots/new_scattered/heatmap_comparison_raw.png')

    # Stats
    print('\n--- Raw Data Statistical Tests ---')
    pvals_raw, sig_raw, diff_raw = run_statistical_tests(
        ari_seq_raw, ari_hier_raw, all_n, all_ov, n_patches_list)

    plot_effect_size_heatmap(
        diff_raw, pvals_raw, sig_raw, all_n, all_ov,
        'Hierarchical \u2212 Sequential ARI Difference \u2014 Raw Data',
        'plots/new_scattered/effect_size_raw.png')

    # 3D embeddings (seed 0)
    plot_3d_embedding_grid(
        emb_seq_raw, ari_seq_raw.mean(axis=2), all_n, all_ov,
        valence, n_patches_subset, 'Sequential',
        f'Quilted Embeddings \u2014 Raw Data ({X_raw.shape[0]} x {X_raw.shape[1]})',
        'plots/new_scattered/embeddings_3d_seq_raw.png')
    plot_3d_embedding_grid(
        emb_hier_raw, ari_hier_raw.mean(axis=2), all_n, all_ov,
        valence, n_patches_subset, 'Hierarchical',
        f'Quilted Embeddings \u2014 Raw Data ({X_raw.shape[0]} x {X_raw.shape[1]})',
        'plots/new_scattered/embeddings_3d_hier_raw.png')

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
        'plots/new_scattered/heatmap_comparison_simulated.png')

    # Stats
    print('\n--- Simulated Data Statistical Tests ---')
    pvals_sim, sig_sim, diff_sim = run_statistical_tests(
        ari_seq_sim, ari_hier_sim, all_n_s, all_ov_s, n_patches_list)

    plot_effect_size_heatmap(
        diff_sim, pvals_sim, sig_sim, all_n_s, all_ov_s,
        'Hierarchical \u2212 Sequential ARI Difference \u2014 Simulated',
        'plots/new_scattered/effect_size_sim.png')

    # 3D embeddings (seed 0)
    plot_3d_embedding_grid(
        emb_seq_sim, ari_seq_sim.mean(axis=2), all_n_s, all_ov_s,
        labels_sim, n_patches_subset, 'Sequential',
        f'Quilted Embeddings \u2014 Simulated ({X_sim.shape[0]} x {X_sim.shape[1]})',
        'plots/new_scattered/embeddings_3d_seq_sim.png')
    plot_3d_embedding_grid(
        emb_hier_sim, ari_hier_sim.mean(axis=2), all_n_s, all_ov_s,
        labels_sim, n_patches_subset, 'Hierarchical',
        f'Quilted Embeddings \u2014 Simulated ({X_sim.shape[0]} x {X_sim.shape[1]})',
        'plots/new_scattered/embeddings_3d_hier_sim.png')

    # ── Line comparison ───────────────────────────────────────────────
    plot_ari_line_comparison(
        ari_seq_raw, ari_hier_raw, bl_raw,
        ari_seq_sim, ari_hier_sim, bl_sim,
        n_patches_list, overlap_list,
        'plots/new_scattered/ari_line_comparison.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
