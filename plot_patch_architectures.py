#!/usr/bin/env python3
"""Generate patch architecture visualizations using the new scattered-block patches."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hierarchical_quilting import generate_patches

Path('plots/new_scattered').mkdir(parents=True, exist_ok=True)

n_patches_list = [3, 5, 8, 12]
overlap_list = [0.10, 0.20, 0.40, 0.80]
n_rows = len(n_patches_list)
n_cols = len(overlap_list)

# Load simulated data shape
sim = np.load('simulated.npz')
M_sim, N_sim = sim['data'].shape

# Raw data shape (from README: 129603 x 121)
M_raw, N_raw = 129603, 121

datasets = [
    ('Raw Data', M_raw, N_raw, 'plots/new_scattered/patch_architectures_raw.png'),
    ('Simulated 3-blob', M_sim, N_sim, 'plots/new_scattered/patch_architectures_simulated.png'),
]

for ds_name, ds_M, ds_N, save_path in datasets:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for row, n_p in enumerate(n_patches_list):
        for col, ov in enumerate(overlap_list):
            ax = axes[row, col]
            p_list = generate_patches(
                shape=(ds_M, ds_N), n_patches=n_p, overlap_frac=ov,
                rng=np.random.default_rng(42),
            )
            mask = np.zeros((ds_M, ds_N), dtype=np.int8)
            for p in p_list:
                mask[np.ix_(p['row_idx'], p['col_idx'])] = 1

            ax.imshow(mask, aspect='auto', cmap='Greys_r', vmin=0, vmax=1)
            ax.set_title(f'{n_p} patches, {ov:.0%} overlap\nobs: {mask.mean():.1%}',
                         fontsize=10)
            if col == 0:
                ax.set_ylabel('Sample index')
            if row == n_rows - 1:
                ax.set_xlabel('Feature index')

    fig.suptitle(f'Patch Architectures — {ds_name} ({ds_M} x {ds_N})\n'
                 '(scattered row blocks, contiguous column ranges)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')
