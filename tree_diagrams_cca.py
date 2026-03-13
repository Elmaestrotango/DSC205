#!/usr/bin/env python3
"""
tree_diagrams_cca.py

Generate two separate tree/chain diagrams for 60-patch CCA alignment
on the simulated dataset, with node colors and labels showing the
fraction of original column information preserved at each merge step.

Outputs:
  plots/sequential_chain_60p_cca.png  — circular chain diagram
  plots/hierarchical_tree_60p_cca.png — binary tree diagram
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from hierarchical_quilting import generate_patches, greedy_patch_ordering, _greedy_pairing

N_PATCHES = 60
OVERLAP_FRAC = 0.60
SEED = 42
RANK = 3

# ── Load simulated data ─────────────────────────────────────────────
sim = np.load('simulated.npz')
X_full = sim['data']
labels_true = sim['labels']
N_total, N_cols = X_full.shape
K = len(np.unique(labels_true))


# ── Helpers ──────────────────────────────────────────────────────────
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


def _mat_sqrt_inv(C):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs * (1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _text_color(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.55 else 'black'


REG = 1e-8

# ── Baseline ─────────────────────────────────────────────────────────
X_pca = PCA(n_components=RANK).fit_transform(X_full)
baseline_ci = column_info(X_pca, X_full)
print(f"Baseline column-info (full PCA, rank {RANK}): {baseline_ci:.4f}")

# ── Generate patches ─────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
patches_data = generate_patches((N_total, N_cols), N_PATCHES,
                                overlap_frac=OVERLAP_FRAC, rng=rng)

# ================================================================
# SEQUENTIAL CCA: instrumented with column-info at each step
# ================================================================
print(f"\n=== Sequential CCA alignment ({N_PATCHES} patches, "
      f"{OVERLAP_FRAC:.0%} overlap) ===")

ordering = greedy_patch_ordering(patches_data, M=N_total)
U_tilde = np.zeros((N_total, RANK))
covered = np.zeros(N_total, dtype=bool)

# Init first patch
p0 = patches_data[ordering[0]]
U, S, Vt = np.linalg.svd(X_full[np.ix_(p0['row_idx'], p0['col_idx'])],
                          full_matrices=False)
U_tilde[p0['row_idx']] = U[:, :RANK]
covered[p0['row_idx']] = True

seq_ci = []
seq_overlaps = [0]
covered_rows = np.where(covered)[0]
ci = column_info(U_tilde[covered_rows], X_full[covered_rows])
seq_ci.append(min(ci / baseline_ci, 1.0))
print(f"  Init: col-info={seq_ci[-1]:.3f}, covered={covered.sum()}")

for step in range(1, len(ordering)):
    m = ordering[step]
    p = patches_data[m]
    row_idx, col_idx = p['row_idx'], p['col_idx']
    U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)],
                              full_matrices=False)
    U_r = U[:, :RANK]

    is_cov = covered[row_idx]
    ov_global = row_idx[is_cov]
    new_global = row_idx[~is_cov]
    n_ov = len(ov_global)

    if n_ov < RANK:
        U_tilde[row_idx] = U_r
        covered[row_idx] = True
        covered_rows = np.where(covered)[0]
        ci = column_info(U_tilde[covered_rows], X_full[covered_rows])
        seq_ci.append(min(ci / baseline_ci, 1.0))
        seq_overlaps.append(n_ov)
        continue

    ov_local = np.searchsorted(row_idx, ov_global)
    covered_idx = np.where(covered)[0]

    # CCA
    U_a_ov = U_tilde[ov_global]
    U_b_ov = U_r[ov_local]
    mu_a, mu_b = U_a_ov.mean(0), U_b_ov.mean(0)
    A_c, B_c = U_a_ov - mu_a, U_b_ov - mu_b

    C_aa = A_c.T @ A_c / (n_ov - 1) + REG * np.eye(RANK)
    C_bb = B_c.T @ B_c / (n_ov - 1) + REG * np.eye(RANK)
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
    covered_rows = np.where(covered)[0]
    ci = column_info(U_tilde[covered_rows], X_full[covered_rows])
    seq_ci.append(min(ci / baseline_ci, 1.0))
    seq_overlaps.append(n_ov)
    if step % 10 == 0 or step == len(ordering) - 1:
        print(f"  Step {step:2d}: overlap={n_ov:5d}, "
              f"col-info={seq_ci[-1]:.3f}, covered={covered.sum()}")

# ================================================================
# HIERARCHICAL CCA: instrumented with column-info at each merge
# ================================================================
print(f"\n=== Hierarchical CCA alignment ({N_PATCHES} patches) ===")


def _init_nodes(patches_data, X_full, r):
    nodes = []
    for p in patches_data:
        row_idx, col_idx = p['row_idx'], p['col_idx']
        U, S, Vt = np.linalg.svd(X_full[np.ix_(row_idx, col_idx)],
                                  full_matrices=False)
        nodes.append({
            'U_dense': U[:, :r].copy(),
            'row_list': np.asarray(row_idx, dtype=np.int64),
        })
    return nodes


nodes = _init_nodes(patches_data, X_full, RANK)

node_id = {i: i for i in range(N_PATCHES)}
node_info = {}

# Leaf column-info
for i, nd in enumerate(nodes):
    ci = column_info(nd['U_dense'], X_full[nd['row_list']])
    n_c = len(patches_data[i]['col_idx'])
    node_info[i] = {
        'ci': min(ci / baseline_ci, 1.0),
        'children': None,
        'label': f'P{i+1}',
        'n_rows': len(nd['row_list']),
        'n_cols': n_c,
    }

print(f"  Leaves: {N_PATCHES} patches, "
      f"mean col-info={np.mean([node_info[i]['ci'] for i in range(N_PATCHES)]):.3f}")

next_id = N_PATCHES
level = 0

while len(nodes) > 1:
    pairs, unpaired_idx = _greedy_pairing(nodes, N_total)
    next_level = []
    new_ids = []

    for i, j in pairs:
        nd_a, nd_b = nodes[i], nodes[j]
        aid, bid = node_id[i], node_id[j]
        rows_a, rows_b = nd_a['row_list'], nd_b['row_list']
        overlap = np.intersect1d(rows_a, rows_b, assume_unique=True)
        n_ov = len(overlap)

        if n_ov >= RANK:
            ov_a = np.searchsorted(rows_a, overlap)
            ov_b = np.searchsorted(rows_b, overlap)

            U_a_ov = nd_a['U_dense'][ov_a]
            U_b_ov = nd_b['U_dense'][ov_b]
            mu_a, mu_b = U_a_ov.mean(0), U_b_ov.mean(0)
            A_c, B_c = U_a_ov - mu_a, U_b_ov - mu_b

            C_aa = A_c.T @ A_c / (n_ov - 1) + REG * np.eye(RANK)
            C_bb = B_c.T @ B_c / (n_ov - 1) + REG * np.eye(RANK)
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

        # Merge: A-only, overlap averaged, B-only
        new_in_b = np.setdiff1d(rows_b, rows_a, assume_unique=True)
        n_a, n_new_b = len(rows_a), len(new_in_b)
        U_cat = np.empty((n_a + n_new_b, RANK))
        U_cat[:n_a] = U_a_cca
        if n_ov >= RANK:
            U_cat[ov_a] = 0.5 * (U_a_cca[ov_a] + U_b_cca[ov_b])
        if n_new_b > 0:
            U_cat[n_a:] = U_b_cca[np.searchsorted(rows_b, new_in_b)]

        union = np.concatenate([rows_a, new_in_b])
        order = np.argsort(union)
        merged = {'U_dense': U_cat[order], 'row_list': union[order]}

        ci = column_info(merged['U_dense'], X_full[merged['row_list']])
        node_info[next_id] = {
            'ci': min(ci / baseline_ci, 1.0),
            'children': (aid, bid),
            'label': f'L{level}',
            'n_rows': len(merged['row_list']),
            'n_overlap': n_ov,
        }

        next_level.append(merged)
        new_ids.append(next_id)
        next_id += 1

    if unpaired_idx is not None:
        next_level.append(nodes[unpaired_idx])
        new_ids.append(node_id[unpaired_idx])

    n_merges = len(pairs)
    ci_vals = [node_info[new_ids[k]]['ci'] for k in range(n_merges)]
    print(f"  Level {level}: {n_merges} merges, "
          f"mean col-info={np.mean(ci_vals):.3f}"
          f"{', +1 unpaired' if unpaired_idx is not None else ''}")

    node_id = {i: nid for i, nid in enumerate(new_ids)}
    nodes = next_level
    level += 1

root_id = node_id[0]
print(f"\nRoot: id={root_id}, col-info={node_info[root_id]['ci']:.3f}")


# ================================================================
# PLOT 1: Sequential CCA — circular chain
# ================================================================
print("\nGenerating sequential CCA circular chain plot...")

cmap = plt.cm.RdYlGn
n_nodes = len(seq_ci)

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_aspect('equal')
ax.axis('off')

radius = 7.5
node_r = 0.32
angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_nodes,
                     endpoint=False)
xs = radius * np.cos(angles)
ys = radius * np.sin(angles)

# Draw edges
for i in range(n_nodes - 1):
    dx = xs[i + 1] - xs[i]
    dy = ys[i + 1] - ys[i]
    d = np.sqrt(dx ** 2 + dy ** 2)
    shrink = (node_r + 0.03) / d
    ax.annotate(
        '', xy=(xs[i + 1] - dx * shrink, ys[i + 1] - dy * shrink),
        xytext=(xs[i] + dx * shrink, ys[i] + dy * shrink),
        arrowprops=dict(arrowstyle='->', color='#555', lw=1.5,
                        connectionstyle='arc3,rad=0.05'))

# Draw nodes
for i, (x, y, ci_val) in enumerate(zip(xs, ys, seq_ci)):
    rgba = cmap(ci_val)
    tc = _text_color(rgba)
    circle = plt.Circle((x, y), node_r, fc=rgba, ec='k', lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, f'{ci_val:.0%}', ha='center', va='center', fontsize=9,
            fontweight='bold', color=tc, zorder=6)

    # Outer label: overlap count (every 5th node to avoid clutter)
    if i == 0 or i % 5 == 0 or i == n_nodes - 1:
        lx = (radius + node_r + 0.4) * np.cos(angles[i])
        ly = (radius + node_r + 0.4) * np.sin(angles[i])
        if i == 0:
            lbl = 'Init'
        else:
            lbl = f'ov={seq_overlaps[i]}'
        ax.text(lx, ly, lbl, ha='center', va='center', fontsize=8,
                fontweight='bold', color='#333')

ax.set_xlim(-radius - 1.5, radius + 1.5)
ax.set_ylim(-radius - 1.5, radius + 1.5)
ax.set_title(f'Sequential CCA alignment — {N_PATCHES} patches, '
             f'{OVERLAP_FRAC:.0%} overlap\n'
             f'Node color/label = column info preserved',
             fontsize=16, fontweight='bold', pad=18)

fig.tight_layout()
fig.savefig('plots/sequential_chain_60p_cca.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved plots/sequential_chain_60p_cca.png")

# ================================================================
# PLOT 2: Hierarchical CCA — binary tree
# ================================================================
print("Generating hierarchical CCA tree plot...")


def _tree_layout(nid, node_info, x_counter=None):
    if x_counter is None:
        x_counter = [0]
    pos = {}
    nfo = node_info[nid]
    if nfo['children'] is None:
        pos[nid] = (x_counter[0], 0)
        x_counter[0] += 1
    else:
        left, right = nfo['children']
        pos.update(_tree_layout(left, node_info, x_counter))
        pos.update(_tree_layout(right, node_info, x_counter))
        lx, ld = pos[left]
        rx, rd = pos[right]
        depth = max(ld, rd) + 1
        pos[nid] = ((lx + rx) / 2, depth)
    return pos


raw_pos = _tree_layout(root_id, node_info)
# Scale: wider for 60 leaves
pos = {nid: (x * 0.9, y * 2.0) for nid, (x, y) in raw_pos.items()}

max_depth = max(y for _, y in pos.values())

fig, ax = plt.subplots(figsize=(max(30, N_PATCHES * 0.6), max(10, max_depth * 1.5 + 2)))
ax.axis('off')

node_r_h = 0.32

# Edges
for nid, nfo in node_info.items():
    if nfo['children'] is not None:
        px, py = pos[nid]
        for child_id in nfo['children']:
            cx, cy = pos[child_id]
            ax.plot([px, cx], [py, cy], color='#888', lw=1.5, zorder=1)

# Nodes
for nid in pos:
    x, y = pos[nid]
    nfo = node_info[nid]
    val = min(nfo['ci'], 1.0)
    rgba = cmap(val)
    tc = _text_color(rgba)

    circle = plt.Circle((x, y), node_r_h, fc=rgba, ec='k', lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, f'{val:.0%}', ha='center', va='center', fontsize=8,
            fontweight='bold', color=tc, zorder=6)

    if nfo['children'] is None:
        # Leaf label: show every patch
        ax.text(x, y - node_r_h - 0.12, nfo['label'],
                ha='center', va='top', fontsize=6, fontweight='bold',
                color='#333')
    else:
        ov_txt = f"ov={nfo.get('n_overlap', '?')}"
        ax.text(x, y + node_r_h + 0.1, ov_txt,
                ha='center', va='bottom', fontsize=7,
                fontweight='bold', color='#555')

all_x = [x for x, _ in pos.values()]
all_y = [y for _, y in pos.values()]
ax.set_xlim(min(all_x) - 1.0, max(all_x) + 1.0)
ax.set_ylim(min(all_y) - 0.8, max(all_y) + 1.2)
ax.set_aspect('equal')

ax.set_title(f'Hierarchical CCA alignment — {N_PATCHES} patches, '
             f'{OVERLAP_FRAC:.0%} overlap\n'
             f'Node color/label = column info preserved',
             fontsize=16, fontweight='bold', pad=18)

fig.tight_layout()
fig.savefig('plots/hierarchical_tree_60p_cca.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved plots/hierarchical_tree_60p_cca.png")

print("\nDone.")
