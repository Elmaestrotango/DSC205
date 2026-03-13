#!/usr/bin/env python3
"""Imputation Baseline Comparison — 3-way comparison.

Compares sequential (lstsq chain), hierarchical (lstsq tree), and a naive
mean-imputation baseline across the same grid used in the other comparison
scripts.

The imputation baseline asks: "Is the quilting alignment procedure actually
better than just filling in missing values and running PCA + k-means?"

Usage:
    python imputation_comparison.py

Outputs:
    plots/heatmap_imputation_raw.png
    plots/heatmap_imputation_simulated.png
    plots/ari_line_imputation.png
    plots/effect_size_imputation_raw.png
    plots/effect_size_imputation_sim.png
    plots/embeddings_3d_impute_raw.png
    plots/embeddings_3d_impute_sim.png
"""

import copy
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

from hierarchical_quilting import (
    generate_patches,
    sequential_quilting,
    full_data_baseline,
    _greedy_pairing,
    _merge_nodes,
    plot_3d_embedding_grid,
)
from procrustes_quilting import _init_nodes, _hierarchical_from_nodes


# ── Imputation baseline ──────────────────────────────────────────────────

def imputation_baseline(patches_data, X_full, r, K):
    """Mean-impute unobserved entries, then PCA + k-means.

    Given patches that each observe a subset of (rows, cols):
    1. Build an observation mask from the patches
    2. Set unobserved entries to NaN
    3. Replace NaN with column means of observed values
    4. Run PCA(r) + k-means(K) on the imputed matrix
    """
    M, N = X_full.shape
    observed = np.zeros((M, N), dtype=bool)
    for p in patches_data:
        observed[np.ix_(p['row_idx'], p['col_idx'])] = True

    X_masked = np.where(observed, X_full, np.nan)
    col_means = np.nanmean(X_masked, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    X_imputed = np.where(np.isnan(X_masked), col_means, X_masked)

    pca = PCA(n_components=r)
    emb = pca.fit_transform(X_imputed)
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(emb)
    return labels, emb


# ── 3-way comparison sweep ───────────────────────────────────────────────

def run_comparison_sweep_imputation(X_data, labels_true, shape, K, r,
                                     n_patches_list, overlap_list,
                                     dataset_name, n_replicates=3, seeds=None):
    """Run sequential, hier-lstsq, and imputation across the full grid.

    Returns
    -------
    ari_seq, ari_hier, ari_imp : ndarray (n_rows, n_cols, n_replicates)
    ari_baseline : float
    all_n, all_ov : list
    emb_seq, emb_hier, emb_imp : dict (i,j) -> ndarray  (seed-0 embeddings)
    """
    if seeds is None:
        seeds = list(range(n_replicates))

    all_n = [0] + n_patches_list
    all_ov = list(overlap_list)
    n_rows, n_cols = len(all_n), len(all_ov)

    ari_seq  = np.zeros((n_rows, n_cols, n_replicates))
    ari_hier = np.zeros((n_rows, n_cols, n_replicates))
    ari_imp  = np.zeros((n_rows, n_cols, n_replicates))
    emb_seq, emb_hier, emb_imp = {}, {}, {}

    # Baseline (row 0)
    baseline_ari, baseline_emb = full_data_baseline(X_data, labels_true, K, r)
    ari_seq[0, :, :]  = baseline_ari
    ari_hier[0, :, :] = baseline_ari
    ari_imp[0, :, :]  = baseline_ari
    for j in range(n_cols):
        emb_seq[(0, j)]  = baseline_emb
        emb_hier[(0, j)] = baseline_emb
        emb_imp[(0, j)]  = baseline_emb
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

                # Hierarchical lstsq
                nodes = _init_nodes(p_list, X_data, r)
                pred_h, U_h = _hierarchical_from_nodes(
                    nodes, X_data.shape[0], r, K, _merge_nodes)
                ari_hier[i, j, rep] = adjusted_rand_score(labels_true, pred_h)

                # Imputation baseline
                pred_i, U_i = imputation_baseline(p_list, X_data, r, K)
                ari_imp[i, j, rep] = adjusted_rand_score(labels_true, pred_i)

                if rep == 0:
                    emb_seq[(i, j)]  = U_s
                    emb_hier[(i, j)] = U_h
                    emb_imp[(i, j)]  = U_i

                run_count += 1

            tag = f'{n_p}p_{int(ov * 100):02d}ov'
            s_m = ari_seq[i, j, :].mean()
            h_m = ari_hier[i, j, :].mean()
            i_m = ari_imp[i, j, :].mean()
            elapsed = time.time() - t0
            eta = elapsed / run_count * (total - run_count)
            print(f'  [{run_count:>4d}/{total}] {tag:>10s}  '
                  f'seq={s_m:.4f}  hier={h_m:.4f}  impute={i_m:.4f}  '
                  f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

    return (ari_seq, ari_hier, ari_imp, baseline_ari,
            all_n, all_ov, emb_seq, emb_hier, emb_imp)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_triple_heatmaps(ari_seq_mean, ari_hier_mean, ari_imp_mean,
                          ari_seq_std, ari_hier_std, ari_imp_std,
                          all_n, all_ov, title, save_path):
    """3-panel ARI heatmap: sequential / hier-lstsq / imputation."""
    vmax = max(0.3, max(ari_seq_mean.max(), ari_hier_mean.max(),
                        ari_imp_mean.max()) * 1.1)

    fig = plt.figure(figsize=(34, 12))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.25)

    for idx, (a_mean, a_std, subtitle) in enumerate(zip(
            [ari_seq_mean, ari_hier_mean, ari_imp_mean],
            [ari_seq_std, ari_hier_std, ari_imp_std],
            ['Sequential (lstsq)', 'Hierarchical (lstsq)', 'Mean Imputation'])):
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


def plot_ari_line_comparison(ari_seq_raw, ari_hier_raw, ari_imp_raw, bl_raw,
                              ari_seq_sim, ari_hier_sim, ari_imp_sim, bl_sim,
                              n_patches_list, overlap_list, save_path):
    """Line plots: ARI vs n_patches, 3 methods + baseline, 2 rows x 5 cols."""
    all_ov = list(overlap_list)
    ov_to_j = {ov: j for j, ov in enumerate(all_ov)}
    n_rep = ari_seq_raw.shape[2]

    fig, axes = plt.subplots(2, len(all_ov),
                              figsize=(4.5 * len(all_ov), 9), sharey='row')

    for row_idx, (a_s, a_h, a_i, bl, ds) in enumerate([
            (ari_seq_raw, ari_hier_raw, ari_imp_raw, bl_raw, 'Raw Data'),
            (ari_seq_sim, ari_hier_sim, ari_imp_sim, bl_sim, 'Simulated')]):

        for col_idx, ov in enumerate(all_ov):
            ax = axes[row_idx, col_idx]
            j = ov_to_j[ov]

            x = np.array(n_patches_list)
            for a_arr, fmt, clr, lbl in [
                    (a_s, 'o-',  '#e41a1c', 'Sequential'),
                    (a_h, 's--', '#377eb8', 'Hier-lstsq'),
                    (a_i, '^-.', '#ff7f00', 'Imputation')]:
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


def plot_effect_size_pair(diff_is, sig_is, diff_ih, sig_ih,
                           all_n, all_ov, title, save_path):
    """Two side-by-side effect-size heatmaps (imputation vs sequential,
    imputation vs hierarchical)."""
    d_is = diff_is[1:, :]
    d_ih = diff_ih[1:, :]
    s_is = sig_is[1:, :]
    s_ih = sig_ih[1:, :]
    row_labels = [str(n) for n in all_n[1:]]
    col_labels = [f'{ov:.0%}' for ov in all_ov]

    vabs = max(0.05, max(np.abs(d_is).max(), np.abs(d_ih).max()) * 1.1)

    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)

    for idx, (d, s, sub) in enumerate(zip(
            [d_is, d_ih], [s_is, s_ih],
            ['Imputation \u2212 Sequential', 'Imputation \u2212 Hierarchical'])):
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


# ── Statistical tests ────────────────────────────────────────────────────

def run_statistical_tests_3way(ari_seq, ari_hier, ari_imp,
                                all_n, all_ov, n_patches_list):
    """Paired t-tests for 3 pairwise comparisons with Bonferroni correction.

    Returns dict with keys 'imp_vs_seq', 'imp_vs_hier', 'hier_vs_seq', each
    a tuple (pvalues, significant, diff_mean).
    """
    n_rows, n_cols, n_rep = ari_seq.shape
    n_grid = len(n_patches_list) * n_cols
    n_tests = 3 * n_grid  # 3 comparisons x grid points
    alpha_corr = 0.05 / n_tests

    results = {}
    comparisons = [
        ('imp_vs_seq',  ari_imp,  ari_seq,  'Imputation vs Sequential'),
        ('imp_vs_hier', ari_imp,  ari_hier, 'Imputation vs Hierarchical'),
        ('hier_vs_seq', ari_hier, ari_seq,  'Hierarchical vs Sequential'),
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


# ── Main ─────────────────────────────────────────────────────────────────

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
    print(f'\n=== Raw Data Imputation Sweep ({len(n_patches_list)} x '
          f'{len(overlap_list)} x {n_replicates} replicates) ===')
    (ari_s_r, ari_h_r, ari_i_r, bl_r, all_n, all_ov,
     emb_s_r, emb_h_r, emb_i_r) = run_comparison_sweep_imputation(
        X_raw, valence, X_raw.shape, K_raw, r_raw,
        n_patches_list, overlap_list, 'raw',
        n_replicates=n_replicates, seeds=seeds)

    plot_triple_heatmaps(
        ari_s_r.mean(2), ari_h_r.mean(2), ari_i_r.mean(2),
        ari_s_r.std(2),  ari_h_r.std(2),  ari_i_r.std(2),
        all_n, all_ov,
        f'Imputation Baseline Comparison \u2014 Raw Data '
        f'({X_raw.shape[0]} x {X_raw.shape[1]}, n={n_replicates})',
        'plots/new_scattered/heatmap_imputation_raw.png')

    print('\n--- Raw Data Statistical Tests ---')
    res_raw = run_statistical_tests_3way(
        ari_s_r, ari_h_r, ari_i_r, all_n, all_ov, n_patches_list)

    _, sig_is_r, diff_is_r = res_raw['imp_vs_seq']
    _, sig_ih_r, diff_ih_r = res_raw['imp_vs_hier']
    plot_effect_size_pair(
        diff_is_r, sig_is_r, diff_ih_r, sig_ih_r,
        all_n, all_ov,
        'ARI Difference: Imputation vs Quilting \u2014 Raw Data',
        'plots/new_scattered/effect_size_imputation_raw.png')

    plot_3d_embedding_grid(
        emb_i_r, ari_i_r.mean(2), all_n, all_ov,
        valence, n_patches_subset, 'Mean Imputation',
        f'Imputed Embeddings \u2014 Raw Data '
        f'({X_raw.shape[0]} x {X_raw.shape[1]})',
        'plots/new_scattered/embeddings_3d_impute_raw.png')

    # ── Simulated sweep ───────────────────────────────────────────────
    print(f'\n=== Simulated Data Imputation Sweep ({len(n_patches_list)} x '
          f'{len(overlap_list)} x {n_replicates} replicates) ===')
    (ari_s_s, ari_h_s, ari_i_s, bl_s, all_n_s, all_ov_s,
     emb_s_s, emb_h_s, emb_i_s) = run_comparison_sweep_imputation(
        X_sim, labels_sim, X_sim.shape, K_sim, r_sim,
        n_patches_list, overlap_list, 'simulated',
        n_replicates=n_replicates, seeds=seeds)

    plot_triple_heatmaps(
        ari_s_s.mean(2), ari_h_s.mean(2), ari_i_s.mean(2),
        ari_s_s.std(2),  ari_h_s.std(2),  ari_i_s.std(2),
        all_n_s, all_ov_s,
        f'Imputation Baseline Comparison \u2014 Simulated '
        f'({X_sim.shape[0]} x {X_sim.shape[1]}, n={n_replicates})',
        'plots/new_scattered/heatmap_imputation_simulated.png')

    print('\n--- Simulated Data Statistical Tests ---')
    res_sim = run_statistical_tests_3way(
        ari_s_s, ari_h_s, ari_i_s, all_n_s, all_ov_s, n_patches_list)

    _, sig_is_s, diff_is_s = res_sim['imp_vs_seq']
    _, sig_ih_s, diff_ih_s = res_sim['imp_vs_hier']
    plot_effect_size_pair(
        diff_is_s, sig_is_s, diff_ih_s, sig_ih_s,
        all_n_s, all_ov_s,
        'ARI Difference: Imputation vs Quilting \u2014 Simulated',
        'plots/new_scattered/effect_size_imputation_sim.png')

    plot_3d_embedding_grid(
        emb_i_s, ari_i_s.mean(2), all_n_s, all_ov_s,
        labels_sim, n_patches_subset, 'Mean Imputation',
        f'Imputed Embeddings \u2014 Simulated '
        f'({X_sim.shape[0]} x {X_sim.shape[1]})',
        'plots/new_scattered/embeddings_3d_impute_sim.png')

    # ── Line comparison ───────────────────────────────────────────────
    plot_ari_line_comparison(
        ari_s_r, ari_h_r, ari_i_r, bl_r,
        ari_s_s, ari_h_s, ari_i_s, bl_s,
        n_patches_list, overlap_list,
        'plots/new_scattered/ari_line_imputation.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
