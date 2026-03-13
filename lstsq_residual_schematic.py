#!/usr/bin/env python3
"""
lstsq_residual_schematic.py

Schematic illustrating how least-squares alignment loses non-overlap
information in the residuals, compared with CCA which preserves both
input spaces.

Layout:
  Row 0: Alignment chain diagrams (lstsq vs CCA, nodes colored by
         fraction of original cluster information preserved)
  Row 1: Information preservation bars (preserved vs lost)
  Row 2: 2D scatter plots (uniform axis limits across all panels)

Saves → plots/lstsq_residual_schematic.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── Geometry ─────────────────────────────────────────────────────────
ov_dir = np.array([1.0, 0.15])
ov_dir /= np.linalg.norm(ov_dir)
ov_perp = np.array([-ov_dir[1], ov_dir[0]])

c1_true = 2.0 * ov_perp
c2_true = -2.0 * ov_perp

n_ov = 10
n_ex = 14
σ = 0.28
σ_ov_major = 0.5
σ_ov_minor = 0.12

# ── True positions in shared space ───────────────────────────────────
ov_along = np.random.uniform(-σ_ov_major, σ_ov_major, n_ov)
ov_across = np.random.randn(n_ov) * σ_ov_minor
overlap_true = np.outer(ov_along, ov_dir) + np.outer(ov_across, ov_perp)

a_only_true = np.vstack([
    np.random.randn(n_ex // 2, 2) * σ + c1_true,
    np.random.randn(n_ex // 2, 2) * σ + c2_true,
])
b_only_true = np.vstack([
    np.random.randn(n_ex // 2, 2) * σ + c1_true,
    np.random.randn(n_ex // 2, 2) * σ + c2_true,
])
labels_ex = np.array([0] * (n_ex // 2) + [1] * (n_ex // 2))

# ── Patch B's SVD frame (rotation + anisotropic scaling) ────────────
θ = 1.1
R = np.array([[np.cos(θ), -np.sin(θ)],
              [np.sin(θ),  np.cos(θ)]])
S_mat = np.diag([1.5, 0.5])
T_B = R @ S_mat

noise = 0.08
overlap_A = overlap_true + np.random.randn(n_ov, 2) * noise
a_only_A  = a_only_true  + np.random.randn(n_ex, 2) * noise
overlap_B = overlap_true @ T_B.T + np.random.randn(n_ov, 2) * noise
b_only_B  = b_only_true  @ T_B.T + np.random.randn(n_ex, 2) * noise

# ── Lstsq alignment ─────────────────────────────────────────────────
G, _, _, _ = np.linalg.lstsq(overlap_B, overlap_A, rcond=None)
overlap_B_lstsq = overlap_B @ G
b_only_lstsq    = b_only_B  @ G

# ── CCA alignment ───────────────────────────────────────────────────
def _mat_sqrt_inv(M):
    w, v = np.linalg.eigh(M)
    return v @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-10, None))) @ v.T


def cca_align(A_ov, B_ov, A_all, B_all):
    mu_a, mu_b = A_ov.mean(0), B_ov.mean(0)
    Ac, Bc = A_ov - mu_a, B_ov - mu_b
    n = len(A_ov)
    reg = 1e-6 * np.eye(2)
    Caa = Ac.T @ Ac / (n - 1) + reg
    Cbb = Bc.T @ Bc / (n - 1) + reg
    Cab = Ac.T @ Bc / (n - 1)
    Wa_isq = _mat_sqrt_inv(Caa)
    Wb_isq = _mat_sqrt_inv(Cbb)
    P, s, Qt = np.linalg.svd(Wa_isq @ Cab @ Wb_isq, full_matrices=False)
    Wa = Wa_isq @ P
    Wb = Wb_isq @ Qt.T
    return (A_all - mu_a) @ Wa, (B_all - mu_b) @ Wb, s


a_all = np.vstack([overlap_A, a_only_A])
b_all = np.vstack([overlap_B, b_only_B])
a_cca, b_cca, canon_corr = cca_align(overlap_A, overlap_B, a_all, b_all)

# Standardize CCA output so whitening doesn't blow up scale
combined_cca = np.vstack([a_cca, b_cca])
for d in range(2):
    mu = combined_cca[:, d].mean()
    sd = combined_cca[:, d].std() + 1e-12
    a_cca[:, d] = (a_cca[:, d] - mu) / sd
    b_cca[:, d] = (b_cca[:, d] - mu) / sd

for d in range(2):
    if np.corrcoef(a_all[:, d], a_cca[:, d])[0, 1] < 0:
        a_cca[:, d] *= -1
        b_cca[:, d] *= -1

ov_a_cca, ex_a_cca = a_cca[:n_ov], a_cca[n_ov:]
ov_b_cca, ex_b_cca = b_cca[:n_ov], b_cca[n_ov:]

# ── Cluster separability metric ──────────────────────────────────────
def cluster_sep(X, labels):
    """Between-cluster variance / total variance."""
    mu = X.mean(0)
    total = np.sum((X - mu)**2)
    between = 0.0
    for l in np.unique(labels):
        mask = labels == l
        mu_l = X[mask].mean(0)
        between += mask.sum() * np.sum((mu_l - mu)**2)
    return between / total if total > 1e-12 else 0.0


sep_vals = [
    cluster_sep(a_only_A, labels_ex),
    cluster_sep(b_only_B, labels_ex),
    cluster_sep(b_only_lstsq, labels_ex),
    cluster_sep(ex_b_cca, labels_ex),
]

# ── Uniform axis limits for all scatter panels ──────────────────────
all_pts = [
    np.vstack([overlap_A, a_only_A]),
    np.vstack([overlap_B, b_only_B]),
    np.vstack([overlap_A, a_only_A, overlap_B_lstsq, b_only_lstsq,
               b_only_true]),
    np.vstack([ex_a_cca, ex_b_cca, ov_a_cca, ov_b_cca]),
]
max_abs = max(np.abs(pts).max() for pts in all_pts)
lim = max_abs + 0.5

# ── Colors ───────────────────────────────────────────────────────────
COL = {
    'c1': '#e41a1c', 'c2': '#377eb8',
    'ov': '#999999', 'resid': '#ff7f00',
    'preserved': '#4daf4a', 'lost': '#e41a1c',
}
MS, MS_OV = 55, 45
cmap_info = plt.cm.RdYlGn


def ccol(labels):
    return [COL['c1'] if l == 0 else COL['c2'] for l in labels]


# ═════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 13))
gs = gridspec.GridSpec(3, 4, height_ratios=[1.4, 0.3, 3.0],
                       hspace=0.25, wspace=0.22)

# ═══ ROW 0: TREE DIAGRAMS ═══════════════════════════════════════════
ax_tree = fig.add_subplot(gs[0, :])
ax_tree.set_xlim(0, 1)
ax_tree.set_ylim(0, 1)
ax_tree.axis('off')

# Chain layout parameters
n_p = 5
chain_x = np.linspace(0.10, 0.68, n_p)
node_sz = 650

# Illustrative degradation values (compounding for lstsq, stable for CCA)
lstsq_info = [1.00, 0.82, 0.63, 0.45, 0.32]
cca_info   = [1.00, 0.95, 0.90, 0.86, 0.82]


def draw_chain(ax, x_arr, info_arr, y_chain, y_leaf, title):
    """Draw a sequential alignment chain with nodes colored by info
    preservation on the RdYlGn colormap."""
    # Title
    ax.text(x_arr[0] - 0.04, y_chain, title, fontsize=12,
            fontweight='bold', va='center', ha='right')

    for i, (x, val) in enumerate(zip(x_arr, info_arr)):
        # Chain node
        ax.scatter([x], [y_chain], s=node_sz, c=[cmap_info(val)],
                   edgecolors='k', linewidths=1.5, zorder=5)
        # Label above
        lbl = '$P_1$' if i == 0 else f'$M_{{1\\text{{-}}{i+1}}}$'
        ax.text(x, y_chain + 0.065, lbl, ha='center', va='bottom',
                fontsize=9)
        # Percentage below chain node
        ax.text(x, y_chain - 0.065, f'{val:.0%}', ha='center', va='top',
                fontsize=8, fontweight='bold',
                color='#333' if val > 0.5 else '#aa0000')

        # Arrow to next chain node
        if i < len(x_arr) - 1:
            ax.annotate(
                '', xy=(x_arr[i + 1] - 0.032, y_chain),
                xytext=(x + 0.032, y_chain),
                arrowprops=dict(arrowstyle='->', color='k', lw=1.5))

    # Leaf (incoming patch) nodes below each merge node
    for i in range(1, len(x_arr)):
        x = x_arr[i]
        ax.scatter([x], [y_leaf], s=node_sz * 0.6, c=[cmap_info(1.0)],
                   edgecolors='k', linewidths=1.0, zorder=5)
        ax.text(x, y_leaf - 0.055, f'$P_{i + 1}$', ha='center',
                va='top', fontsize=8)
        # Arrow from leaf up to chain
        ax.annotate(
            '', xy=(x, y_chain - 0.035),
            xytext=(x, y_leaf + 0.03),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.0))

    # "result" label after last node
    ax.annotate(
        f'  result\n  ({info_arr[-1]:.0%})',
        xy=(x_arr[-1] + 0.035, y_chain),
        fontsize=10, va='center', fontweight='bold',
        color='#333' if info_arr[-1] > 0.5 else '#aa0000')


draw_chain(ax_tree, chain_x, lstsq_info,
           y_chain=0.78, y_leaf=0.58, title='Lstsq\n(sequential)')
draw_chain(ax_tree, chain_x, cca_info,
           y_chain=0.30, y_leaf=0.10, title='CCA\n(sequential)')

# Colorbar for tree
sm = plt.cm.ScalarMappable(cmap=cmap_info, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.82, 0.72, 0.012, 0.16])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Info preserved', fontsize=9)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
cbar.ax.tick_params(labelsize=7)

# ═══ ROW 1: INFORMATION PRESERVATION BARS ════════════════════════════
bar_titles = ['Patch A (raw)', 'Patch B (raw)',
              'Lstsq aligned', 'CCA aligned']
for col in range(4):
    ax = fig.add_subplot(gs[1, col])
    preserved = sep_vals[col]
    lost = 1.0 - preserved

    ax.barh([0], [preserved], color=COL['preserved'], height=0.6,
            edgecolor='k', linewidth=0.5)
    ax.barh([0], [lost], left=[preserved], color=COL['lost'],
            height=0.6, edgecolor='k', linewidth=0.5, alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0%', '50%', '100%'], fontsize=7)
    ax.set_title(bar_titles[col], fontsize=10, fontweight='bold', pad=2)

    # Label inside preserved region
    if preserved > 0.15:
        ax.text(preserved / 2, 0, f'{preserved:.0%}', ha='center',
                va='center', fontsize=9, fontweight='bold', color='white')
    # Label inside lost region
    if lost > 0.15:
        ax.text(preserved + lost / 2, 0, f'{lost:.0%}', ha='center',
                va='center', fontsize=8, fontweight='bold', color='white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Bar legend
bar_legend_ax = fig.add_subplot(gs[1, :])
bar_legend_ax.axis('off')
bar_handles = [
    Line2D([0], [0], marker='s', color='w', markersize=10,
           markerfacecolor=COL['preserved'], markeredgecolor='k',
           label='Cluster info preserved'),
    Line2D([0], [0], marker='s', color='w', markersize=10,
           markerfacecolor=COL['lost'], alpha=0.5, markeredgecolor='k',
           label='Cluster info lost'),
]

# ═══ ROW 2: SCATTER PLOTS ═══════════════════════════════════════════
ax_plots = [fig.add_subplot(gs[2, i]) for i in range(4)]

# Panel 1: Patch A
ax = ax_plots[0]
ax.scatter(*overlap_A.T, c=COL['ov'], s=MS_OV, marker='o',
           edgecolors='k', linewidths=0.5, zorder=3)
ax.scatter(*a_only_A.T, c=ccol(labels_ex), s=MS, marker='s',
           edgecolors='k', linewidths=0.5, zorder=3)
ax.set_title('Patch A  (SVD space)', fontsize=13, fontweight='bold')
ax.set_xlabel('$u_1$', fontsize=11)
ax.set_ylabel('$u_2$', fontsize=11)
ax.annotate('overlap samples\n(poor $\\perp$ coverage)',
            xy=(0, 0), fontsize=8, fontstyle='italic', color='#555555',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc',
                      alpha=0.85))

# Panel 2: Patch B
ax = ax_plots[1]
ax.scatter(*overlap_B.T, c=COL['ov'], s=MS_OV, marker='o',
           edgecolors='k', linewidths=0.5, zorder=3)
ax.scatter(*b_only_B.T, c=ccol(labels_ex), s=MS, marker='D',
           edgecolors='k', linewidths=0.5, zorder=3)
ax.set_title("Patch B  (SVD space)", fontsize=13, fontweight='bold')
ax.set_xlabel("$u_1'$", fontsize=11)
ax.set_ylabel("$u_2'$", fontsize=11)

# Panel 3: Lstsq
ax = ax_plots[2]
ax.scatter(*a_only_A.T, c=ccol(labels_ex), s=MS, marker='s',
           edgecolors='k', linewidths=0.5, zorder=2, alpha=0.3)
ax.scatter(*overlap_A.T, c=COL['ov'], s=MS_OV, marker='o',
           edgecolors='k', linewidths=0.4, zorder=3, alpha=0.4)
ax.scatter(*overlap_B_lstsq.T, c=COL['ov'], s=MS_OV, marker='o',
           edgecolors='k', linewidths=0.5, zorder=3)
ax.scatter(*b_only_lstsq.T, c=ccol(labels_ex), s=MS, marker='D',
           edgecolors='k', linewidths=0.5, zorder=4)
# Ghost ideal positions
ax.scatter(*b_only_true.T, c=ccol(labels_ex), s=MS, marker='D',
           edgecolors=COL['resid'], linewidths=1.8, alpha=0.3, zorder=2)
# Residual arrows
for i in range(n_ex):
    ax.annotate('', xy=b_only_true[i], xytext=b_only_lstsq[i],
                arrowprops=dict(arrowstyle='->', color=COL['resid'],
                                lw=1.4, alpha=0.7))
ax.set_title('Lstsq: B $\\rightarrow$ A space\nresiduals = lost cluster info',
             fontsize=13, fontweight='bold')
ax.set_xlabel('$u_1$', fontsize=11)
ax.set_ylabel('$u_2$', fontsize=11)
ax.annotate('$\\perp$ structure\ncollapsed',
            xy=(b_only_lstsq[:, 0].mean(), b_only_lstsq[:, 1].mean()),
            xytext=(b_only_lstsq[:, 0].mean() + lim * 0.35,
                    b_only_lstsq[:, 1].mean() + lim * 0.25),
            fontsize=9, fontstyle='italic', color=COL['resid'],
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COL['resid'], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=COL['resid'], alpha=0.9))

# Panel 4: CCA
ax = ax_plots[3]
ax.scatter(*ex_a_cca.T, c=ccol(labels_ex), s=MS, marker='s',
           edgecolors='k', linewidths=0.5, zorder=3)
ax.scatter(*ov_a_cca.T, c=COL['ov'], s=MS_OV, marker='o',
           edgecolors='k', linewidths=0.5, zorder=3, alpha=0.5)
ax.scatter(*ex_b_cca.T, c=ccol(labels_ex), s=MS, marker='D',
           edgecolors='k', linewidths=0.5, zorder=4)
ax.scatter(*ov_b_cca.T, c=COL['ov'], s=MS_OV, marker='v',
           edgecolors='k', linewidths=0.5, zorder=4, alpha=0.5)
ax.set_title('CCA: shared subspace\nboth spaces preserved',
             fontsize=13, fontweight='bold')
ax.set_xlabel('$z_1$', fontsize=11)
ax.set_ylabel('$z_2$', fontsize=11)
ax.annotate(f'canon. corr = {canon_corr[0]:.2f}, {canon_corr[1]:.2f}',
            xy=(0.03, 0.03), xycoords='axes fraction',
            fontsize=7, fontstyle='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc',
                      alpha=0.85))

# ── Uniform axes for all scatter panels ──────────────────────────────
for ax in ax_plots:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)

# ── Shared scatter legend ────────────────────────────────────────────
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COL['ov'],
           markeredgecolor='k', markersize=8, label='Overlap'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COL['c1'],
           markeredgecolor='k', markersize=8, label='Cluster 1'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COL['c2'],
           markeredgecolor='k', markersize=8, label='Cluster 2'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#ccc',
           markeredgecolor=COL['resid'], markeredgewidth=1.8, markersize=8,
           alpha=0.5, label='Ideal position'),
    Line2D([0], [0], color=COL['resid'], lw=1.8,
           label='Residual (lost info)'),
    Line2D([0], [0], marker='s', color='w', markersize=8,
           markerfacecolor=COL['preserved'], markeredgecolor='k',
           label='Info preserved (bar)'),
    Line2D([0], [0], marker='s', color='w', markersize=8,
           markerfacecolor=COL['lost'], markeredgecolor='k', alpha=0.5,
           label='Info lost (bar)'),
]
fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=9,
           bbox_to_anchor=(0.5, -0.01), frameon=True,
           fancybox=True, edgecolor='#ccc')

plt.savefig('plots/lstsq_residual_schematic.png', dpi=150,
            bbox_inches='tight')
plt.close()
print('Saved → plots/lstsq_residual_schematic.png')
