#!/usr/bin/env python3
"""Procrustes Cluster Quilting — 3-way comparison.

Compares sequential (lstsq chain), hierarchical (lstsq tree), and
hierarchical (Procrustes tree) quilting across the same grid used in
hierarchical_quilting.py.

The only algorithmic change: replace the unconstrained least-squares
alignment G with an orthogonal Procrustes rotation Q in the hierarchical
merge step.  Everything else (patch generation, SVD, greedy pairing,
tree structure) is identical.

Usage:
    python procrustes_quilting.py

Outputs:
    plots/heatmap_3way_raw.png
    plots/heatmap_3way_simulated.png
    plots/ari_line_3way.png
    plots/embeddings_3d_proc_raw.png
    plots/embeddings_3d_proc_sim.png
    plots/effect_size_3way_raw.png
    plots/effect_size_3way_sim.png
"""

import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from hierarchical_quilting import (
    generate_patches,
    sequential_quilting,
    full_data_baseline,
    _greedy_pairing,
    _merge_nodes,
    plot_3d_embedding_grid,
)


# ── Procrustes merge ──────────────────────────────────────────────────────

def _merge_nodes_procrustes(node_a, node_b, r):
    """Merge two nodes by aligning node_b onto node_a via orthogonal Procrustes.

    Identical to _merge_nodes except the alignment step:
      lstsq:      G = argmin_G ||U_b G - U_a||_F   (unconstrained)
      Procrustes: Q = argmin_{Q^T Q=I} ||U_b Q - U_a||_F  (orthogonal)
    """
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

        # orthogonal_procrustes(A, B) → R minimising ||A R - B||_F, R^T R = I
        Q, _ = orthogonal_procrustes(U_b_ov, U_a_ov)
        U_b_aligned = node_b['U_dense'] @ Q
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


# ── Shared helpers ─────────────────────────────────────────────────────────

def _init_nodes(patches_data, X_full, r):
    """Per-patch SVD — shared by both hierarchical methods."""
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
    return nodes


def _hierarchical_from_nodes(nodes, N_total, r, K, merge_fn):
    """Run hierarchical quilting from pre-computed nodes with the given merge function."""
    work = copy.deepcopy(nodes)

    while len(work) > 1:
        pairs, unpaired = _greedy_pairing(work)
        next_level = []

        for i, j in pairs:
            if len(work[i]['row_list']) >= len(work[j]['row_list']):
                merged = merge_fn(work[i], work[j], r)
            else:
                merged = merge_fn(work[j], work[i], r)
            next_level.append(merged)

        if unpaired is not None:
            next_level.append(work[unpaired])
        work = next_level

    final = work[0]
    U_tilde = np.zeros((N_total, r))
    U_tilde[final['row_list'], :] = final['U_dense']

    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(U_tilde)
    return labels, U_tilde


# ── 3-way comparison sweep ─────────────────────────────────────────────────

def run_comparison_sweep_3way(X_data, labels_true, shape, K, r,
                               n_patches_list, overlap_list, dataset_name,
                               n_replicates=3, seeds=None):
    """Run sequential, hier-lstsq, and hier-Procrustes across the full grid.

    Returns
    -------
    ari_seq, ari_hier, ari_proc : ndarray (n_rows, n_cols, n_replicates)
    ari_baseline : float
    all_n, all_ov : list
    emb_seq, emb_hier, emb_proc : dict (i,j) -> ndarray  (seed-0 embeddings)
    """
    if seeds is None:
        seeds = list(range(n_replicates))

    all_n = [0] + n_patches_list
    all_ov = list(overlap_list)
    n_rows, n_cols = len(all_n), len(all_ov)

    ari_seq  = np.zeros((n_rows, n_cols, n_replicates))
    ari_hier = np.zeros((n_rows, n_cols, n_replicates))
    ari_proc = np.zeros((n_rows, n_cols, n_replicates))
    emb_seq, emb_hier, emb_proc = {}, {}, {}

    # Baseline (row 0)
    baseline_ari, baseline_emb = full_data_baseline(X_data, labels_true, K, r)
    ari_seq[0, :, :]  = baseline_ari
    ari_hier[0, :, :] = baseline_ari
    ari_proc[0, :, :] = baseline_ari
    for j in range(n_cols):
        emb_seq[(0, j)]  = baseline_emb
        emb_hier[(0, j)] = baseline_emb
        emb_proc[(0, j)] = baseline_emb
    print(f'  [baseline]  ARI={baseline_ari:.4f}  (PCA r={r} + k-means)')

    total = len(n_patches_list) * len(all_ov) * n_replicates
    run_count = 0
    t0 = time.time()

    for i, n_p in enumerate(n_patches_list, start=1):
        for j, ov in enumerate(all_ov):
            for rep, seed in enumerate(seeds):
                p_list = generate_patches(
                    shape=shape, n_patches=n_p, overlap_frac=ov,
                    rng=np.random.default_rng(seed))

                # Sequential
                pred_s, U_s = sequential_quilting(p_list, X_data, r=r, K=K)
                ari_seq[i, j, rep] = adjusted_rand_score(labels_true, pred_s)

                # Shared SVD for both hierarchical variants
                nodes = _init_nodes(p_list, X_data, r)

                pred_h, U_h = _hierarchical_from_nodes(
                    nodes, X_data.shape[0], r, K, _merge_nodes)
                ari_hier[i, j, rep] = adjusted_rand_score(labels_true, pred_h)

                pred_p, U_p = _hierarchical_from_nodes(
                    nodes, X_data.shape[0], r, K, _merge_nodes_procrustes)
                ari_proc[i, j, rep] = adjusted_rand_score(labels_true, pred_p)

                if rep == 0:
                    emb_seq[(i, j)]  = U_s
                    emb_hier[(i, j)] = U_h
                    emb_proc[(i, j)] = U_p

                run_count += 1

            tag = f'{n_p}p_{int(ov * 100):02d}ov'
            s_m = ari_seq[i, j, :].mean()
            h_m = ari_hier[i, j, :].mean()
            p_m = ari_proc[i, j, :].mean()
            elapsed = time.time() - t0
            eta = elapsed / run_count * (total - run_count)
            print(f'  [{run_count:>4d}/{total}] {tag:>10s}  '
                  f'seq={s_m:.4f}  hier={h_m:.4f}  proc={p_m:.4f}  '
                  f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

    return (ari_seq, ari_hier, ari_proc, baseline_ari,
            all_n, all_ov, emb_seq, emb_hier, emb_proc)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_triple_heatmaps(ari_seq_mean, ari_hier_mean, ari_proc_mean,
                          ari_seq_std, ari_hier_std, ari_proc_std,
                          all_n, all_ov, title, save_path):
    """3-panel ARI heatmap: sequential / hier-lstsq / hier-Procrustes."""
    vmax = max(0.3, max(ari_seq_mean.max(), ari_hier_mean.max(),
                        ari_proc_mean.max()) * 1.1)

    fig = plt.figure(figsize=(34, 12))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.25)

    for idx, (a_mean, a_std, subtitle) in enumerate(zip(
            [ari_seq_mean, ari_hier_mean, ari_proc_mean],
            [ari_seq_std, ari_hier_std, ari_proc_std],
            ['Sequential (lstsq)', 'Hierarchical (lstsq)', 'Hierarchical (Procrustes)'])):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(a_mean, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(all_ov)))
        ax.set_xticklabels([f'{ov:.0%}' for ov in all_ov])
        ax.set_yticks(range(len(all_n)))
        ax.set_yticklabels([str(n) if n > 0 else 'Full data' for n in all_n])
        ax.set_xlabel('Sample overlap fraction')
        ax.set_ylabel('Number of patches')
        ax.set_title(subtitle)

        for i in range(len(all_n)):
            for j in range(len(all_ov)):
                val = a_mean[i, j]
                sd = a_std[i, j]
                color = 'white' if val > vmax * 0.55 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)
                if sd > 0:
                    ax.text(j, i + 0.3, f'\u00b1{sd:.3f}', ha='center',
                            va='center', fontsize=6, color=color, alpha=0.7)

    cbar_ax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im, cax=cbar_ax, label='Adjusted Rand Index')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_ari_line_comparison_3way(ari_seq_raw, ari_hier_raw, ari_proc_raw, bl_raw,
                                   ari_seq_sim, ari_hier_sim, ari_proc_sim, bl_sim,
                                   n_patches_list, overlap_list, save_path):
    """Line plots: ARI vs n_patches, 3 methods + baseline, 2 rows x 5 cols."""
    all_ov = list(overlap_list)
    ov_to_j = {ov: j for j, ov in enumerate(all_ov)}
    n_rep = ari_seq_raw.shape[2]

    fig, axes = plt.subplots(2, len(all_ov),
                              figsize=(4.5 * len(all_ov), 9), sharey='row')

    for row_idx, (a_s, a_h, a_p, bl, ds) in enumerate([
            (ari_seq_raw, ari_hier_raw, ari_proc_raw, bl_raw, 'Raw Data'),
            (ari_seq_sim, ari_hier_sim, ari_proc_sim, bl_sim, 'Simulated')]):

        for col_idx, ov in enumerate(all_ov):
            ax = axes[row_idx, col_idx]
            j = ov_to_j[ov]

            x = np.array(n_patches_list)
            for a_arr, fmt, clr, lbl in [
                    (a_s, 'o-',  '#e41a1c', 'Sequential'),
                    (a_h, 's--', '#377eb8', 'Hier-lstsq'),
                    (a_p, '^-.', '#4daf4a', 'Hier-Procrustes')]:
                vals = np.array([a_arr[i, j, :]
                                 for i in range(1, len(n_patches_list) + 1)])
                m = vals.mean(axis=1)
                sem = vals.std(axis=1) / np.sqrt(n_rep)
                ax.plot(x, m, fmt, color=clr, label=lbl, linewidth=2, markersize=4)
                ax.fill_between(x, m - sem, m + sem, color=clr, alpha=0.15)

            ax.axhline(bl, color='#999999', ls=':', lw=1.5,
                       label='Full-data baseline')
            ax.set_xlabel('Number of patches')
            ax.set_title(f'{ds} \u2014 {ov:.0%} overlap')
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) if v % 3 == 0 else '' for v in x],
                                fontsize=7)
            if col_idx == 0:
                ax.set_ylabel('ARI')
            if row_idx == 0 and col_idx == len(all_ov) - 1:
                ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_effect_size_pair(diff_hl, sig_hl, diff_hp, sig_hp,
                           all_n, all_ov, title, save_path):
    """Two side-by-side effect-size heatmaps (lstsq−seq and Procrustes−seq)."""
    d_hl = diff_hl[1:, :]
    d_hp = diff_hp[1:, :]
    s_hl = sig_hl[1:, :]
    s_hp = sig_hp[1:, :]
    row_labels = [str(n) for n in all_n[1:]]
    col_labels = [f'{ov:.0%}' for ov in all_ov]

    vabs = max(0.05, max(np.abs(d_hl).max(), np.abs(d_hp).max()) * 1.1)

    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)

    for idx, (d, s, sub) in enumerate(zip(
            [d_hl, d_hp], [s_hl, s_hp],
            ['Hier-lstsq \u2212 Sequential', 'Hier-Procrustes \u2212 Sequential'])):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(d, cmap='RdBu', aspect='auto', vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel('Sample overlap fraction')
        ax.set_ylabel('Number of patches')
        ax.set_title(sub)

        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = d[i, j]
                color = 'white' if abs(val) > vabs * 0.55 else 'black'
                text = f'{val:+.3f}'
                if s[i, j]:
                    text += ' *'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=7, fontweight='bold', color=color)

    cbar_ax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im, cax=cbar_ax, label='ARI difference')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Statistical tests ──────────────────────────────────────────────────────

def run_statistical_tests_3way(ari_seq, ari_hier, ari_proc,
                                all_n, all_ov, n_patches_list):
    """Paired t-tests for 3 pairwise comparisons with Bonferroni correction.

    Returns dict with keys 'hl_vs_seq', 'hp_vs_seq', 'hp_vs_hl', each a
    tuple (pvalues, significant, diff_mean).
    """
    n_rows, n_cols, n_rep = ari_seq.shape
    n_grid = len(n_patches_list) * n_cols
    n_tests = 3 * n_grid  # 3 comparisons x grid points
    alpha_corr = 0.05 / n_tests

    results = {}
    comparisons = [
        ('hl_vs_seq', ari_hier, ari_seq,  'Hier-lstsq vs Sequential'),
        ('hp_vs_seq', ari_proc, ari_seq,  'Hier-Procrustes vs Sequential'),
        ('hp_vs_hl',  ari_proc, ari_hier, 'Hier-Procrustes vs Hier-lstsq'),
    ]

    for key, ari_a, ari_b, label in comparisons:
        pvals = np.full((n_rows, n_cols), np.nan)
        diff = np.zeros((n_rows, n_cols))

        for i in range(1, n_rows):
            for j in range(n_cols):
                a_vals = ari_a[i, j, :]
                b_vals = ari_b[i, j, :]
                diff[i, j] = a_vals.mean() - b_vals.mean()
                if np.allclose(a_vals, b_vals):
                    pvals[i, j] = 1.0
                else:
                    _, p = ttest_rel(a_vals, b_vals)
                    pvals[i, j] = p

        sig = pvals < alpha_corr
        results[key] = (pvals, sig, diff)

        print(f'\n  {label} (Bonferroni alpha={alpha_corr:.6f}, n={n_rep}):')
        print(f'  {"Config":>12s}  {"diff":>10s}  {"p-value":>10s}  {"sig":>4s}')
        print(f'  {"-"*12}  {"-"*10}  {"-"*10}  {"-"*4}')
        for i in range(1, n_rows):
            for j in range(n_cols):
                tag = f'{all_n[i]}p_{int(all_ov[j]*100):02d}ov'
                d = diff[i, j]
                p = pvals[i, j]
                s = '*' if sig[i, j] else ''
                if abs(d) > 0.01:
                    print(f'  {tag:>12s}  {d:>+10.4f}  {p:>10.4f}  {s:>4s}')

        n_sig = int(np.nansum(sig))
        print(f'  {n_sig}/{n_grid} significant after Bonferroni')

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    Path('plots/new_scattered').mkdir(parents=True, exist_ok=True)

    n_patches_list = list(range(1, 16))
    overlap_list = [0.10, 0.20, 0.40, 0.80]
    n_replicates = 5
    seeds = list(range(n_replicates))
    n_patches_subset = [1, 3, 5, 8, 12, 15]

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
    print(f'\n=== Raw Data 3-Way Sweep ({len(n_patches_list)} x '
          f'{len(overlap_list)} x {n_replicates} replicates) ===')
    (ari_s_r, ari_h_r, ari_p_r, bl_r, all_n, all_ov,
     emb_s_r, emb_h_r, emb_p_r) = run_comparison_sweep_3way(
        X_raw, valence, X_raw.shape, K_raw, r_raw,
        n_patches_list, overlap_list, 'raw',
        n_replicates=n_replicates, seeds=seeds)

    plot_triple_heatmaps(
        ari_s_r.mean(2), ari_h_r.mean(2), ari_p_r.mean(2),
        ari_s_r.std(2),  ari_h_r.std(2),  ari_p_r.std(2),
        all_n, all_ov,
        f'3-Way ARI Comparison \u2014 Raw Data '
        f'({X_raw.shape[0]} x {X_raw.shape[1]}, n={n_replicates})',
        'plots/new_scattered/heatmap_3way_raw.png')

    print('\n--- Raw Data Statistical Tests ---')
    res_raw = run_statistical_tests_3way(
        ari_s_r, ari_h_r, ari_p_r, all_n, all_ov, n_patches_list)

    _, sig_hl_r, diff_hl_r = res_raw['hl_vs_seq']
    _, sig_hp_r, diff_hp_r = res_raw['hp_vs_seq']
    plot_effect_size_pair(
        diff_hl_r, sig_hl_r, diff_hp_r, sig_hp_r,
        all_n, all_ov,
        'ARI Difference vs Sequential \u2014 Raw Data',
        'plots/new_scattered/effect_size_3way_raw.png')

    plot_3d_embedding_grid(
        emb_p_r, ari_p_r.mean(2), all_n, all_ov,
        valence, n_patches_subset, 'Hier-Procrustes',
        f'Quilted Embeddings \u2014 Raw Data '
        f'({X_raw.shape[0]} x {X_raw.shape[1]})',
        'plots/new_scattered/embeddings_3d_proc_raw.png')

    # ── Simulated sweep ───────────────────────────────────────────────
    print(f'\n=== Simulated Data 3-Way Sweep ({len(n_patches_list)} x '
          f'{len(overlap_list)} x {n_replicates} replicates) ===')
    (ari_s_s, ari_h_s, ari_p_s, bl_s, all_n_s, all_ov_s,
     emb_s_s, emb_h_s, emb_p_s) = run_comparison_sweep_3way(
        X_sim, labels_sim, X_sim.shape, K_sim, r_sim,
        n_patches_list, overlap_list, 'simulated',
        n_replicates=n_replicates, seeds=seeds)

    plot_triple_heatmaps(
        ari_s_s.mean(2), ari_h_s.mean(2), ari_p_s.mean(2),
        ari_s_s.std(2),  ari_h_s.std(2),  ari_p_s.std(2),
        all_n_s, all_ov_s,
        f'3-Way ARI Comparison \u2014 Simulated '
        f'({X_sim.shape[0]} x {X_sim.shape[1]}, n={n_replicates})',
        'plots/new_scattered/heatmap_3way_simulated.png')

    print('\n--- Simulated Data Statistical Tests ---')
    res_sim = run_statistical_tests_3way(
        ari_s_s, ari_h_s, ari_p_s, all_n_s, all_ov_s, n_patches_list)

    _, sig_hl_s, diff_hl_s = res_sim['hl_vs_seq']
    _, sig_hp_s, diff_hp_s = res_sim['hp_vs_seq']
    plot_effect_size_pair(
        diff_hl_s, sig_hl_s, diff_hp_s, sig_hp_s,
        all_n_s, all_ov_s,
        'ARI Difference vs Sequential \u2014 Simulated',
        'plots/new_scattered/effect_size_3way_sim.png')

    plot_3d_embedding_grid(
        emb_p_s, ari_p_s.mean(2), all_n_s, all_ov_s,
        labels_sim, n_patches_subset, 'Hier-Procrustes',
        f'Quilted Embeddings \u2014 Simulated '
        f'({X_sim.shape[0]} x {X_sim.shape[1]})',
        'plots/new_scattered/embeddings_3d_proc_sim.png')

    # ── Line comparison ───────────────────────────────────────────────
    plot_ari_line_comparison_3way(
        ari_s_r, ari_h_r, ari_p_r, bl_r,
        ari_s_s, ari_h_s, ari_p_s, bl_s,
        n_patches_list, overlap_list,
        'plots/new_scattered/ari_line_3way.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
