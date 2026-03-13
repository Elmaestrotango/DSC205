#!/usr/bin/env python3
"""Alignment fidelity diagnostic: seq-lstsq vs seq-CCA vs hier-lstsq vs hier-CCA.

5-seed replication, 14 patches, 40% overlap, both datasets.

Usage:
    python alignment_fidelity.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from hierarchical_quilting import (
    generate_patches,
    greedy_patch_ordering,
    _greedy_pairing,
)

N_PATCHES = 14
OVERLAP_FRAC = 0.40
SEEDS = [42, 123, 456, 789, 1024]
METHOD_NAMES = ['Seq-lstsq', 'Seq-CCA', 'Hier-lstsq', 'Hier-CCA']
METHOD_COLORS = {'Seq-lstsq': '#e41a1c', 'Seq-CCA': '#ff7f00',
                 'Hier-lstsq': '#377eb8', 'Hier-CCA': '#4daf4a'}


# ── Shared helpers ────────────────────────────────────────────────────────

def _init_nodes(patches_data, X_full, r):
    """Per-patch SVD → list of nodes."""
    nodes = []
    for p in patches_data:
        row_idx, col_idx = p['row_idx'], p['col_idx']
        X_m = X_full[np.ix_(row_idx, col_idx)]
        U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
        nodes.append({
            'U_dense': U[:, :r].copy(),
            'row_list': np.asarray(row_idx, dtype=np.int64),
        })
    return nodes


def _mat_sqrt_inv(C):
    """Inverse square root of a symmetric positive-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs * (1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _finalize(nodes, N_total, r, K):
    """Scatter final node into full embedding, run k-means."""
    final = nodes[0]
    U_tilde = np.zeros((N_total, r))
    U_tilde[final['row_list'], :] = final['U_dense']
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(U_tilde)
    return labels, U_tilde


# ── Instrumented sequential quilting ──────────────────────────────────────

def sequential_instrumented(patches_data, X_full, r):
    M_total = X_full.shape[0]
    ordering = greedy_patch_ordering(patches_data, M=M_total)
    U_tilde = np.zeros((M_total, r))
    covered = np.zeros(M_total, dtype=bool)
    steps = []

    row_idx = patches_data[ordering[0]]['row_idx']
    col_idx = patches_data[ordering[0]]['col_idx']
    U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)], full_matrices=False)
    U_tilde[row_idx, :] = U[:, :r]
    covered[row_idx] = True

    for step in range(1, len(ordering)):
        m = ordering[step]
        row_idx = patches_data[m]['row_idx']
        col_idx = patches_data[m]['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)], full_matrices=False)
        U_r = U[:, :r]

        is_cov = covered[row_idx]
        ov_global = row_idx[is_cov]
        new_global = row_idx[~is_cov]
        n_ov = len(ov_global)

        if n_ov < r:
            U_tilde[row_idx, :] = U_r
            covered[row_idx] = True
            steps.append(dict(step=step, n_overlap=n_ov,
                              residual=np.nan, cond_G=np.nan))
            continue

        ov_local = np.searchsorted(row_idx, ov_global)
        G, _, _, _ = np.linalg.lstsq(
            U_r[ov_local], U_tilde[ov_global], rcond=None)

        fitted = U_r[ov_local] @ G
        resid = np.linalg.norm(fitted - U_tilde[ov_global], 'fro') / n_ov
        sv = np.linalg.svd(G, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else np.inf

        if len(new_global) > 0:
            new_local = np.searchsorted(row_idx, new_global)
            U_tilde[new_global] = U_r[new_local] @ G
        covered[row_idx] = True

        steps.append(dict(step=step, n_overlap=n_ov,
                          residual=resid, cond_G=cond))

    return U_tilde, steps


# ── Instrumented sequential-CCA ──────────────────────────────────────────

def sequential_cca_instrumented(patches_data, X_full, r):
    """Sequential chain quilting using CCA instead of lstsq at each merge."""
    M_total = X_full.shape[0]
    ordering = greedy_patch_ordering(patches_data, M=M_total)
    U_tilde = np.zeros((M_total, r))
    covered = np.zeros(M_total, dtype=bool)
    steps = []
    REG = 1e-8

    row_idx = patches_data[ordering[0]]['row_idx']
    col_idx = patches_data[ordering[0]]['col_idx']
    U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)], full_matrices=False)
    U_tilde[row_idx, :] = U[:, :r]
    covered[row_idx] = True

    for step in range(1, len(ordering)):
        m = ordering[step]
        row_idx = patches_data[m]['row_idx']
        col_idx = patches_data[m]['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)], full_matrices=False)
        U_r = U[:, :r]

        is_cov = covered[row_idx]
        ov_global = row_idx[is_cov]
        new_global = row_idx[~is_cov]
        n_ov = len(ov_global)

        if n_ov < r:
            U_tilde[row_idx, :] = U_r
            covered[row_idx] = True
            steps.append(dict(step=step, n_overlap=n_ov,
                              residual=np.nan, cond_G=np.nan,
                              canon_corr=np.array([])))
            continue

        ov_local = np.searchsorted(row_idx, ov_global)
        covered_idx = np.where(covered)[0]

        # CCA between accumulated (A) and new patch (B) on overlap
        U_a_ov = U_tilde[ov_global]
        U_b_ov = U_r[ov_local]

        mu_a = U_a_ov.mean(axis=0)
        mu_b = U_b_ov.mean(axis=0)
        A_c = U_a_ov - mu_a
        B_c = U_b_ov - mu_b

        C_aa = A_c.T @ A_c / (n_ov - 1) + REG * np.eye(r)
        C_bb = B_c.T @ B_c / (n_ov - 1) + REG * np.eye(r)
        C_ab = A_c.T @ B_c / (n_ov - 1)

        C_aa_isq = _mat_sqrt_inv(C_aa)
        C_bb_isq = _mat_sqrt_inv(C_bb)
        M = C_aa_isq @ C_ab @ C_bb_isq
        P, canon_corr, Qt = np.linalg.svd(M, full_matrices=False)

        W_a = C_aa_isq @ P
        W_b = C_bb_isq @ Qt.T

        # Project accumulated covered rows through W_a
        U_a_cca = (U_tilde[covered_idx] - mu_a) @ W_a
        # Project new patch through W_b
        U_b_cca = (U_r - mu_b) @ W_b

        # Residual on overlap in CCA space
        ov_in_covered = np.searchsorted(covered_idx, ov_global)
        resid = np.linalg.norm(
            U_a_cca[ov_in_covered] - U_b_cca[ov_local], 'fro') / n_ov
        cond = (canon_corr[0] / canon_corr[-1]
                if canon_corr[-1] > 1e-12 else np.inf)

        # Update: re-project all previously covered rows
        U_tilde[covered_idx] = U_a_cca
        # Overlap: average of both projections
        U_tilde[ov_global] = 0.5 * (U_a_cca[ov_in_covered] + U_b_cca[ov_local])
        # New rows from this patch
        if len(new_global) > 0:
            new_local = np.searchsorted(row_idx, new_global)
            U_tilde[new_global] = U_b_cca[new_local]

        covered[row_idx] = True
        steps.append(dict(step=step, n_overlap=n_ov,
                          residual=resid, cond_G=cond,
                          canon_corr=canon_corr))

    return U_tilde, steps


# ── Instrumented hierarchical-lstsq ──────────────────────────────────────

def hier_lstsq_instrumented(patches_data, X_full, r):
    N_total = X_full.shape[0]
    nodes = _init_nodes(patches_data, X_full, r)
    steps = []
    level = 0

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes, N_total)
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
                fitted = child['U_dense'][ov_b] @ G
                resid = np.linalg.norm(fitted - anchor['U_dense'][ov_a], 'fro') / n_ov
                sv = np.linalg.svd(G, compute_uv=False)
                cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else np.inf

                U_b_aligned = child['U_dense'] @ G
            else:
                resid, cond = np.nan, np.nan
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
            steps.append(dict(level=level, n_overlap=n_ov,
                              residual=resid, cond_G=cond))

        if unpaired is not None:
            next_level.append(nodes[unpaired])
        nodes = next_level
        level += 1

    return nodes, steps, N_total


# ── Instrumented hierarchical-CCA ────────────────────────────────────────

def hier_cca_instrumented(patches_data, X_full, r):
    N_total = X_full.shape[0]
    nodes = _init_nodes(patches_data, X_full, r)
    steps = []
    level = 0
    REG = 1e-8  # Tikhonov regularisation for covariance inversion

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes, N_total)
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

                # Center on overlap
                mu_a = U_a_ov.mean(axis=0)
                mu_b = U_b_ov.mean(axis=0)
                A_c = U_a_ov - mu_a
                B_c = U_b_ov - mu_b

                # Covariance matrices (regularised)
                C_aa = A_c.T @ A_c / (n_ov - 1) + REG * np.eye(r)
                C_bb = B_c.T @ B_c / (n_ov - 1) + REG * np.eye(r)
                C_ab = A_c.T @ B_c / (n_ov - 1)

                # CCA via SVD of whitened cross-covariance
                C_aa_isq = _mat_sqrt_inv(C_aa)
                C_bb_isq = _mat_sqrt_inv(C_bb)
                M = C_aa_isq @ C_ab @ C_bb_isq
                P, canon_corr, Qt = np.linalg.svd(M, full_matrices=False)

                W_a = C_aa_isq @ P          # (r, r)
                W_b = C_bb_isq @ Qt.T       # (r, r)

                # Project all samples into CCA space
                U_a_cca = (nd_a['U_dense'] - mu_a) @ W_a
                U_b_cca = (nd_b['U_dense'] - mu_b) @ W_b

                # Residual: disagreement on overlap in CCA space
                resid = np.linalg.norm(
                    U_a_cca[ov_a] - U_b_cca[ov_b], 'fro') / n_ov
                # Condition: ratio of max/min canonical correlation
                cond = (canon_corr[0] / canon_corr[-1]
                        if canon_corr[-1] > 1e-12 else np.inf)
            else:
                resid, cond = np.nan, np.nan
                canon_corr = np.array([])
                # No alignment possible — keep raw embeddings
                mu_a = mu_b = 0.0
                W_a = W_b = np.eye(r)
                U_a_cca = nd_a['U_dense']
                U_b_cca = nd_b['U_dense']
                ov_a = np.searchsorted(rows_a, overlap)
                ov_b = np.searchsorted(rows_b, overlap)

            # Merge: A-only → W_a projection, B-only → W_b, overlap → average
            new_in_a = np.setdiff1d(rows_a, rows_b, assume_unique=True)
            new_in_b = np.setdiff1d(rows_b, rows_a, assume_unique=True)

            # Build merged arrays in global-row order
            union = np.concatenate([rows_a, new_in_b])
            n_a = len(rows_a)
            n_new_b = len(new_in_b)
            U_cat = np.empty((n_a + n_new_b, r))
            U_cat[:n_a] = U_a_cca                # A's full projection
            # Overwrite overlap positions with average
            if n_ov >= r:
                U_cat[ov_a] = 0.5 * (U_a_cca[ov_a] + U_b_cca[ov_b])
            if n_new_b > 0:
                U_cat[n_a:] = U_b_cca[np.searchsorted(rows_b, new_in_b)]

            order = np.argsort(union)
            next_level.append({'U_dense': U_cat[order], 'row_list': union[order]})

            steps.append(dict(level=level, n_overlap=n_ov,
                              residual=resid, cond_G=cond,
                              canon_corr=canon_corr))

        if unpaired is not None:
            next_level.append(nodes[unpaired])
        nodes = next_level
        level += 1

    return nodes, steps, N_total


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_fidelity(seq_steps, seq_cca_steps, lstsq_steps, cca_steps,
                  dataset_name, save_path, seed_label=''):
    """Fidelity diagnostic for one (median) seed."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    seq_x = np.array([s['step'] for s in seq_steps])
    seq_cca_x = np.array([s['step'] for s in seq_cca_steps])

    def _hier_x(steps):
        levels = np.array([s['level'] for s in steps])
        max_lv = levels.max()
        n_seq = len(seq_steps)
        scale = (n_seq - 1) / max_lv if max_lv > 0 else 1.0
        base = levels * scale
        jitter = np.zeros(len(steps))
        for lv in range(max_lv + 1):
            mask = levels == lv
            n = mask.sum()
            if n > 1:
                jitter[mask] = np.linspace(-0.25 * scale, 0.25 * scale, n)
        return base + jitter, levels, max_lv, scale

    lstsq_x, lstsq_levels, max_lv, scale = _hier_x(lstsq_steps)
    cca_x, _, _, _ = _hier_x(cca_steps)
    cca_x += 0.15 * scale

    metrics = [
        ('residual', 'Alignment residual (per sample)', 'Alignment Residual', False),
        ('n_overlap', 'Overlap sample count', 'Shared Samples per Merge', False),
        ('cond_G', 'cond(G) or canon_corr ratio', 'Condition / Correlation Ratio', True),
    ]

    for ax, (key, ylabel, title, use_log) in zip(axes, metrics):
        seq_y = np.array([s[key] for s in seq_steps], dtype=float)
        seq_cca_y = np.array([s[key] for s in seq_cca_steps], dtype=float)
        lstsq_y = np.array([s[key] for s in lstsq_steps], dtype=float)
        cca_y = np.array([s[key] for s in cca_steps], dtype=float)

        ax.plot(seq_x, seq_y, 'o-', color=METHOD_COLORS['Seq-lstsq'],
                label='Seq-lstsq', markersize=5, linewidth=1.5, zorder=3)
        ax.plot(seq_cca_x, seq_cca_y, 's-', color=METHOD_COLORS['Seq-CCA'],
                label='Seq-CCA', markersize=5, linewidth=1.5, zorder=3)
        ax.scatter(lstsq_x, lstsq_y, marker='D', color=METHOD_COLORS['Hier-lstsq'],
                   s=40, label='Hier-lstsq', zorder=4)
        ax.scatter(cca_x, cca_y, marker='^', color=METHOD_COLORS['Hier-CCA'],
                   s=40, label='Hier-CCA', zorder=4)

        for lv in range(max_lv + 1):
            mask = lstsq_levels == lv
            lv_x = lstsq_x[mask]
            if len(lv_x) > 0:
                ax.axvspan(lv_x.min() - 0.3 * scale,
                           lv_x.max() + 0.45 * scale,
                           alpha=0.08, color='#377eb8')

        ax2 = ax.twiny()
        centers = [lstsq_x[lstsq_levels == lv].mean() for lv in range(max_lv + 1)]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(centers)
        ax2.set_xticklabels([f'L{lv}' for lv in range(max_lv + 1)],
                             fontsize=8, color='#377eb8')
        ax2.tick_params(length=0)

        ax.set_xlabel('Sequential chain step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if use_log:
            ax.set_yscale('log')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Alignment Fidelity — {dataset_name} '
                 f'({N_PATCHES} patches, {OVERLAP_FRAC:.0%} overlap'
                 f'{seed_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_canonical_correlations(seq_cca_steps, hier_cca_steps,
                                dataset_name, save_path, seed_label=''):
    """Bar chart of canonical correlations: top = Seq-CCA, bottom = Hier-CCA."""
    fig, (ax_seq, ax_hier) = plt.subplots(2, 1, figsize=(14, 7))

    for ax, steps, method in [
        (ax_seq, seq_cca_steps, 'Seq-CCA'),
        (ax_hier, hier_cca_steps, 'Hier-CCA'),
    ]:
        x_pos = 0
        ticks, labels = [], []
        for s in steps:
            cc = s.get('canon_corr', np.array([]))
            if len(cc) == 0:
                continue
            k = len(cc)
            xs = np.arange(k) + x_pos
            colors = plt.cm.viridis(cc)
            ax.bar(xs, cc, width=0.8, color=colors, edgecolor='k', linewidth=0.3)
            label = f'S{s["step"]}' if 'step' in s else f'L{s["level"]}'
            ticks.append(x_pos + k / 2 - 0.5)
            labels.append(label)
            x_pos += k + 1

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('Canonical correlation')
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color='grey', ls=':', lw=0.8)
        ax.set_title(f'{method}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'CCA Canonical Correlations — {dataset_name} '
                 f'({N_PATCHES}p, {OVERLAP_FRAC:.0%} ov{seed_label})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_ari_bar(all_aris, dataset_name, save_path):
    """Bar chart: mean ARI ± std with individual seed points and pairwise p-values."""
    methods = METHOD_NAMES
    means = np.array([np.mean(all_aris[m]) for m in methods])
    stds = np.array([np.std(all_aris[m], ddof=1) for m in methods])
    colors = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='k', linewidth=0.5, alpha=0.85, zorder=2)

    # Overlay individual seed points
    for i, m in enumerate(methods):
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(all_aris[m]))
        ax.scatter(x[i] + jitter, all_aris[m], color='k', s=25,
                   zorder=3, alpha=0.7, edgecolors='white', linewidths=0.5)

    # Annotate means
    for i, (bar, mu, sd) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mu + sd + 0.008,
                f'{mu:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Pairwise paired t-tests: CCA vs lstsq within each topology
    pairs = [('Seq-lstsq', 'Seq-CCA'), ('Hier-lstsq', 'Hier-CCA'),
             ('Seq-lstsq', 'Hier-lstsq'), ('Seq-CCA', 'Hier-CCA')]
    y_top = means.max() + stds.max() + 0.02
    for pair_idx, (m_a, m_b) in enumerate(pairs):
        a_vals = np.array(all_aris[m_a])
        b_vals = np.array(all_aris[m_b])
        if np.allclose(a_vals, b_vals):
            continue
        _, pval = ttest_rel(a_vals, b_vals)
        i_a, i_b = methods.index(m_a), methods.index(m_b)
        y = y_top + pair_idx * 0.018
        ax.plot([i_a, i_a, i_b, i_b], [y - 0.004, y, y, y - 0.004],
                color='k', linewidth=0.8)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text((i_a + i_b) / 2, y + 0.002,
                f'p={pval:.3f} ({sig})', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_ylim(0, y_top + len(pairs) * 0.018 + 0.03)
    ax.set_title(f'ARI Comparison — {dataset_name}\n'
                 f'({N_PATCHES} patches, {OVERLAP_FRAC:.0%} overlap, '
                 f'n={len(SEEDS)} seeds)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_3d_embeddings(embeddings, pred_labels, true_labels, aris,
                       dataset_name, save_path, seed_label='',
                       n_subsample=800):
    """Side-by-side 3D scatter: one subplot per method, colored by true labels."""
    rng_plot = np.random.default_rng(42)
    unique = np.unique(true_labels)
    K = len(unique)
    cmap = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3'][:K]
    colors = {lab: c for lab, c in zip(unique, cmap)}

    sub_idx = np.concatenate([
        rng_plot.choice(np.where(true_labels == lab)[0],
                        size=min(n_subsample, (true_labels == lab).sum()),
                        replace=False)
        for lab in unique
    ])
    labels_sub = true_labels[sub_idx]

    methods = list(embeddings.keys())
    fig = plt.figure(figsize=(6 * len(methods), 5))

    for col, method in enumerate(methods):
        ax = fig.add_subplot(1, len(methods), col + 1, projection='3d')
        emb = embeddings[method][sub_idx]

        for lab in unique:
            mask = labels_sub == lab
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                       s=3, alpha=0.3, color=colors[lab], label=str(lab))

        ax.set_title(f'{method}\nARI={aris[method]:.3f}', fontsize=11)
        ax.tick_params(labelsize=6)
        if col == len(methods) - 1:
            ax.legend(markerscale=4, fontsize=8, loc='upper right')

    fig.suptitle(f'Quilted Embeddings (true labels) — {dataset_name}\n'
                 f'({N_PATCHES} patches, {OVERLAP_FRAC:.0%} overlap'
                 f'{seed_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────

def _load_raw():
    df = pd.read_csv('trialdf_24sessions.csv', index_col=0)
    meta = ['valence', 'airstart', 'sucstart', 'ms_id',
            'condition', 'inj_site', 'ms_n']
    feat = [c for c in df.columns if c not in meta]
    df = df.loc[~df['valence'].str.contains('CS', na=False)]
    thresh = 0.05 * len(df)
    feat = [c for c in feat if df[c].isna().sum() <= thresh]
    df = df[feat + meta].dropna()
    X = StandardScaler().fit_transform(df[feat].values)
    labels = df['valence'].values
    return X, labels


def _run_one_seed(X, labels, K, r, seed):
    """Run all 4 methods on one patch realization. Returns dict of results."""
    patches = generate_patches(
        X.shape, N_PATCHES, OVERLAP_FRAC, rng=np.random.default_rng(seed))
    N = X.shape[0]

    U_seq, seq_steps = sequential_instrumented(patches, X, r)
    U_seq_cca, seq_cca_steps = sequential_cca_instrumented(patches, X, r)
    nodes_l, lstsq_steps, _ = hier_lstsq_instrumented(patches, X, r)
    labels_l, U_lstsq = _finalize(nodes_l, N, r, K)
    nodes_c, cca_steps, _ = hier_cca_instrumented(patches, X, r)
    labels_c, U_cca = _finalize(nodes_c, N, r, K)

    km1 = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels_seq = km1.fit_predict(U_seq)
    km2 = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels_seq_cca = km2.fit_predict(U_seq_cca)

    aris = {
        'Seq-lstsq': adjusted_rand_score(labels, labels_seq),
        'Seq-CCA': adjusted_rand_score(labels, labels_seq_cca),
        'Hier-lstsq': adjusted_rand_score(labels, labels_l),
        'Hier-CCA': adjusted_rand_score(labels, labels_c),
    }
    return {
        'seed': seed, 'aris': aris,
        'embeddings': {'Seq-lstsq': U_seq, 'Seq-CCA': U_seq_cca,
                       'Hier-lstsq': U_lstsq, 'Hier-CCA': U_cca},
        'pred_labels': {'Seq-lstsq': labels_seq, 'Seq-CCA': labels_seq_cca,
                        'Hier-lstsq': labels_l, 'Hier-CCA': labels_c},
        'steps': {'seq': seq_steps, 'seq_cca': seq_cca_steps,
                  'lstsq': lstsq_steps, 'cca': cca_steps},
    }


def _run_dataset(X, labels, dataset_name):
    K = len(np.unique(labels))
    r = K
    tag = dataset_name.split()[0].lower()

    all_aris = {m: [] for m in METHOD_NAMES}
    seed_results = []

    for i, seed in enumerate(SEEDS):
        print(f'  Seed {seed} ({i + 1}/{len(SEEDS)})...')
        res = _run_one_seed(X, labels, K, r, seed)
        seed_results.append(res)
        for m in METHOD_NAMES:
            all_aris[m].append(res['aris'][m])
        print(f'    ARI: ' + '  '.join(
            f'{m}={res["aris"][m]:.4f}' for m in METHOD_NAMES))

    # Print summary
    print(f'  ── Summary ({dataset_name}) ──')
    for m in METHOD_NAMES:
        vals = all_aris[m]
        print(f'    {m:12s}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}')

    # Pick median seed (by mean ARI across methods)
    mean_per_seed = [np.mean([sr['aris'][m] for m in METHOD_NAMES])
                     for sr in seed_results]
    median_idx = np.argsort(mean_per_seed)[len(SEEDS) // 2]
    med = seed_results[median_idx]
    seed_label = f', median seed={SEEDS[median_idx]}'

    # Plots
    plot_ari_bar(all_aris, dataset_name, f'plots/cca/ari_bar_{tag}.png')

    plot_fidelity(med['steps']['seq'], med['steps']['seq_cca'],
                  med['steps']['lstsq'], med['steps']['cca'],
                  dataset_name, f'plots/cca/alignment_fidelity_{tag}.png',
                  seed_label=seed_label)

    plot_canonical_correlations(
        med['steps']['seq_cca'], med['steps']['cca'],
        dataset_name, f'plots/cca/canon_corr_{tag}.png',
        seed_label=seed_label)

    plot_3d_embeddings(med['embeddings'], med['pred_labels'], labels,
                       med['aris'], dataset_name,
                       f'plots/cca/embeddings_3d_{tag}.png',
                       seed_label=seed_label)


def main():
    Path('plots/cca').mkdir(parents=True, exist_ok=True)

    print('Loading raw data...')
    X_raw, labels_raw = _load_raw()
    print(f'  Raw: {X_raw.shape}, K={len(np.unique(labels_raw))}')
    _run_dataset(X_raw, labels_raw, 'Raw Data')

    print('Loading simulated data...')
    sim = np.load('simulated.npz')
    X_sim, labels_sim = sim['data'], sim['labels']
    print(f'  Sim: {X_sim.shape}, K={len(np.unique(labels_sim))}')
    _run_dataset(X_sim, labels_sim, 'Simulated')

    print('Done.')


if __name__ == '__main__':
    main()
