#!/usr/bin/env python3
"""Proof-of-concept: quantify greedy-locality overlap decay in hierarchical quilting.

Instruments the merge tree for 15 patches (hier-lstsq) and records per-merge
overlap statistics at each tree level.  Produces a single figure showing how
the overlap fraction collapses as the tree deepens.

Usage:
    python -u overlap_decay.py
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from hierarchical_quilting import generate_patches, _greedy_pairing

# ── Instrumented merge tree ───────────────────────────────────────────────

def _run_instrumented_tree(patches_data, X_full, r):
    """Build the hierarchical merge tree, recording stats at every merge.

    Returns a list of dicts, one per merge:
        level, overlap_count, node_a_size, node_b_size, overlap_frac,
        cond_number (of U_b[overlap])
    """
    # Init leaf nodes (patchwise SVD)
    nodes = []
    for p in patches_data:
        X_m = X_full[np.ix_(p['row_idx'], p['col_idx'])]
        U, S, Vt = np.linalg.svd(X_m, full_matrices=False)
        nodes.append({
            'U_dense': U[:, :r].copy(),
            'row_list': np.array(p['row_idx'], dtype=np.int64),
        })

    records = []
    level = 0

    while len(nodes) > 1:
        pairs, unpaired = _greedy_pairing(nodes)
        next_level = []

        for i, j in pairs:
            # Anchor = larger node
            if len(nodes[i]['row_list']) >= len(nodes[j]['row_list']):
                node_a, node_b = nodes[i], nodes[j]
            else:
                node_a, node_b = nodes[j], nodes[i]

            rows_a = set(node_a['row_list'].tolist())
            rows_b = set(node_b['row_list'].tolist())
            overlap = sorted(rows_a & rows_b)

            n_a = len(node_a['row_list'])
            n_b = len(node_b['row_list'])
            ov_count = len(overlap)
            ov_frac = ov_count / min(n_a, n_b) if min(n_a, n_b) > 0 else 0.0

            # Condition number of alignment matrix
            cond = np.nan
            if ov_count >= r:
                g2l_b = {g: l for l, g in enumerate(node_b['row_list'].tolist())}
                local_ov_b = [g2l_b[g] for g in overlap]
                U_b_ov = node_b['U_dense'][local_ov_b, :]
                cond = np.linalg.cond(U_b_ov)

            records.append({
                'level': level,
                'overlap_count': ov_count,
                'node_a_size': n_a,
                'node_b_size': n_b,
                'overlap_frac': ov_frac,
                'cond_number': cond,
            })

            # Actually perform the merge (same as _merge_nodes)
            g2l_a = {g: l for l, g in enumerate(node_a['row_list'].tolist())}
            g2l_b = {g: l for l, g in enumerate(node_b['row_list'].tolist())}

            if ov_count >= r:
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
            n_new = len(new_in_b)
            U_merged = np.empty((n_a + n_new, r), dtype=np.float64)
            U_merged[:n_a, :] = node_a['U_dense']
            if n_new > 0:
                new_local_b = [g2l_b[g] for g in new_in_b]
                U_merged[n_a:, :] = U_b_aligned[new_local_b, :]

            next_level.append({
                'U_dense': U_merged,
                'row_list': np.array(union_rows, dtype=np.int64),
            })

        if unpaired is not None:
            next_level.append(nodes[unpaired])

        nodes = next_level
        level += 1

    return records


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    n_patches = 15
    overlap_levels = [0.10, 0.20, 0.40, 0.80]
    seed = 0

    # Load raw data
    print('Loading raw data...')
    df = pd.read_csv('trialdf_24sessions.csv', index_col=0)
    meta_cols = ['valence', 'airstart', 'sucstart', 'ms_id',
                 'condition', 'inj_site', 'ms_n']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df.loc[~df['valence'].str.contains('CS', na=False)]
    thresh = 0.05 * len(df)
    feature_cols = [c for c in feature_cols if df[c].isna().sum() <= thresh]
    df = df[feature_cols + meta_cols].dropna()
    X_raw = StandardScaler().fit_transform(df[feature_cols].values)
    K = len(np.unique(df['valence'].values))
    r = K
    print(f'  {X_raw.shape[0]} x {X_raw.shape[1]}, K={K}, r={r}')

    # Run instrumented tree for each overlap level
    all_records = {}
    for ov in overlap_levels:
        print(f'  Running {n_patches}p, {ov:.0%} overlap...')
        patches = generate_patches(
            shape=X_raw.shape, n_patches=n_patches, overlap_frac=ov,
            feature_redundancy=0.15, rng=np.random.default_rng(seed))
        records = _run_instrumented_tree(patches, X_raw, r)
        all_records[ov] = records
        for rec in records:
            print(f'    level={rec["level"]}  overlap={rec["overlap_count"]:>5d}  '
                  f'frac={rec["overlap_frac"]:.3f}  '
                  f'sizes=({rec["node_a_size"]}, {rec["node_b_size"]})  '
                  f'cond={rec["cond_number"]:.1f}')

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = {'0.1': '#e41a1c', '0.2': '#377eb8', '0.4': '#4daf4a', '0.8': '#984ea3'}

    for ov, records in all_records.items():
        levels = [r['level'] for r in records]
        ov_fracs = [r['overlap_frac'] for r in records]
        ov_counts = [r['overlap_count'] for r in records]
        conds = [r['cond_number'] for r in records]
        clr = colors[str(ov)]
        label = f'{ov:.0%} overlap'

        # Panel 1: Overlap fraction vs tree level
        axes[0].scatter(levels, ov_fracs, color=clr, s=60, alpha=0.7,
                        edgecolors='k', linewidths=0.5, label=label, zorder=3)

        # Panel 2: Raw overlap count vs tree level
        axes[1].scatter(levels, ov_counts, color=clr, s=60, alpha=0.7,
                        edgecolors='k', linewidths=0.5, label=label, zorder=3)

        # Panel 3: Condition number vs tree level
        axes[2].scatter(levels, conds, color=clr, s=60, alpha=0.7,
                        edgecolors='k', linewidths=0.5, label=label, zorder=3)

    # Format panels
    axes[0].set_xlabel('Tree level (0 = leaf pairs)')
    axes[0].set_ylabel('Overlap fraction\n(|overlap| / min(|A|, |B|))')
    axes[0].set_title('Overlap fraction decays with tree depth')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Tree level (0 = leaf pairs)')
    axes[1].set_ylabel('Shared sample count')
    axes[1].set_title('Absolute overlap at each merge')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Tree level (0 = leaf pairs)')
    axes[2].set_ylabel('Condition number of U_b[overlap]')
    axes[2].set_title('Alignment conditioning worsens at root')
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Hierarchical Quilting — Overlap Decay ({n_patches} patches, raw data)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/overlap_decay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved plots/overlap_decay.png')


if __name__ == '__main__':
    main()
