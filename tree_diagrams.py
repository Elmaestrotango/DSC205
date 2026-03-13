#!/usr/bin/env python3
"""
tree_diagrams.py

Generate two separate tree/chain diagrams for 14-patch lstsq alignment
on the simulated dataset, with node colors and labels showing the
fraction of original column information preserved at each merge step.

Metric: mean R² across all columns when regressing X onto the node's
rank-r embedding, normalized to the full-PCA baseline.

Outputs:
  plots/sequential_chain_14p.png  — circular chain diagram
  plots/hierarchical_tree_14p.png — binary tree diagram
"""

import numpy as np
import matplotlib.pyplot as plt

from hierarchical_quilting import generate_patches, greedy_patch_ordering, _greedy_pairing

N_PATCHES = 14
OVERLAP_FRAC = 0.40
SEED = 42
RANK = 3  # number of clusters in simulated data

# ── Load simulated data ─────────────────────────────────────────────
sim = np.load('simulated.npz')
X_full = sim['data']
labels_true = sim['labels']
N_total, N_cols = X_full.shape
K = len(np.unique(labels_true))


# ── Column-information metric ───────────────────────────────────────
def column_info(U_node, X_sub):
    """Mean R² across all columns when regressing X_sub onto U_node.

    For each column j, fits  X_sub[:, j] ≈ U_node @ b_j  via OLS and
    computes R².  Returns the mean R² over all columns, clipped to [0, 1].
    """
    B, _, _, _ = np.linalg.lstsq(U_node, X_sub, rcond=None)
    X_hat = U_node @ B
    ss_res = np.sum((X_sub - X_hat) ** 2, axis=0)
    ss_tot = np.sum((X_sub - X_sub.mean(axis=0)) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2 = np.zeros(X_sub.shape[1])
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    r2 = np.clip(r2, 0.0, 1.0)
    return float(r2.mean())


# ── Full-data baseline ──────────────────────────────────────────────
from sklearn.decomposition import PCA
X_pca = PCA(n_components=RANK).fit_transform(X_full)
baseline_ci = column_info(X_pca, X_full)
print(f"Baseline column-info (full PCA, rank {RANK}): {baseline_ci:.4f}")

# ── Generate patches ─────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
patches_data = generate_patches((N_total, N_cols), N_PATCHES,
                                overlap_frac=OVERLAP_FRAC, rng=rng)

# ================================================================
# SEQUENTIAL: instrumented with column-info at each step
# ================================================================
print("\n=== Sequential alignment ===")
ordering = greedy_patch_ordering(patches_data, M=N_total)
U_tilde = np.zeros((N_total, RANK))
covered = np.zeros(N_total, dtype=bool)

# Init first patch
p0 = patches_data[ordering[0]]
U, S, Vt = np.linalg.svd(X_full[np.ix_(p0['row_idx'], p0['col_idx'])],
                          full_matrices=False)
U_tilde[p0['row_idx']] = U[:, :RANK]
covered[p0['row_idx']] = True

seq_ci = []       # column-info fraction at each step (normalized)
seq_overlaps = [0] # overlap count (0 for init)
covered_rows = np.where(covered)[0]
ci = column_info(U_tilde[covered_rows], X_full[covered_rows])
seq_ci.append(min(ci / baseline_ci, 1.0))
n_cols_p0 = len(p0['col_idx'])
print(f"  Init P0: cols={n_cols_p0}/{N_cols} ({n_cols_p0/N_cols:.0%}), "
      f"col-info={seq_ci[-1]:.3f}, covered={covered.sum()}")

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
    else:
        ov_local = np.searchsorted(row_idx, ov_global)
        G, _, _, _ = np.linalg.lstsq(U_r[ov_local], U_tilde[ov_global],
                                     rcond=None)
        if len(new_global) > 0:
            new_local = np.searchsorted(row_idx, new_global)
            U_tilde[new_global] = U_r[new_local] @ G

    covered[row_idx] = True
    covered_rows = np.where(covered)[0]
    ci = column_info(U_tilde[covered_rows], X_full[covered_rows])
    seq_ci.append(min(ci / baseline_ci, 1.0))
    seq_overlaps.append(n_ov)
    print(f"  Step {step:2d}: merge P{m}, overlap={n_ov:5d}, "
          f"col-info={seq_ci[-1]:.3f}, covered={covered.sum()}")

# ================================================================
# HIERARCHICAL: instrumented with column-info at each merge
# ================================================================
print("\n=== Hierarchical alignment ===")


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

# Track the full merge history as a tree
# Each node gets an ID.  Leaves are 0..13, merges get IDs 14, 15, ...
node_id = {i: i for i in range(N_PATCHES)}
node_info = {}

# Compute initial leaf column-info
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
    print(f"  Leaf P{i+1}: cols={n_c}/{N_cols} ({n_c/N_cols:.0%}), "
          f"col-info={node_info[i]['ci']:.3f}")

next_id = N_PATCHES
level = 0

while len(nodes) > 1:
    pairs, unpaired_idx = _greedy_pairing(nodes, N_total)
    next_level = []
    new_ids = []

    for i, j in pairs:
        if len(nodes[i]['row_list']) >= len(nodes[j]['row_list']):
            anchor, child = nodes[i], nodes[j]
            aid, cid = node_id[i], node_id[j]
        else:
            anchor, child = nodes[j], nodes[i]
            aid, cid = node_id[j], node_id[i]

        rows_a = anchor['row_list']
        rows_b = child['row_list']
        overlap = np.intersect1d(rows_a, rows_b, assume_unique=True)
        n_ov = len(overlap)

        if n_ov >= RANK:
            ov_a = np.searchsorted(rows_a, overlap)
            ov_b = np.searchsorted(rows_b, overlap)
            G, _, _, _ = np.linalg.lstsq(
                child['U_dense'][ov_b], anchor['U_dense'][ov_a], rcond=None)
            U_b_aligned = child['U_dense'] @ G
        else:
            U_b_aligned = child['U_dense']

        new_in_b = np.setdiff1d(rows_b, rows_a, assume_unique=True)
        n_a, n_new = len(rows_a), len(new_in_b)
        U_cat = np.empty((n_a + n_new, RANK))
        U_cat[:n_a] = anchor['U_dense']
        if n_new > 0:
            U_cat[n_a:] = U_b_aligned[np.searchsorted(rows_b, new_in_b)]
        union = np.concatenate([rows_a, new_in_b])
        order = np.argsort(union)
        merged = {'U_dense': U_cat[order], 'row_list': union[order]}

        # Measure column-info of merged node
        ci = column_info(merged['U_dense'], X_full[merged['row_list']])
        node_info[next_id] = {
            'ci': min(ci / baseline_ci, 1.0),
            'children': (aid, cid),
            'label': f'L{level}',
            'n_rows': len(merged['row_list']),
            'n_overlap': n_ov,
        }
        print(f"  Level {level}: merge {node_info[aid]['label']}+"
              f"{node_info[cid]['label']} → id={next_id}, "
              f"overlap={n_ov}, col-info={node_info[next_id]['ci']:.3f}")

        next_level.append(merged)
        new_ids.append(next_id)
        next_id += 1

    # Handle unpaired node (promoted)
    if unpaired_idx is not None:
        next_level.append(nodes[unpaired_idx])
        new_ids.append(node_id[unpaired_idx])

    node_id = {i: nid for i, nid in enumerate(new_ids)}
    nodes = next_level
    level += 1

root_id = node_id[0]
print(f"\nRoot: id={root_id}, col-info={node_info[root_id]['ci']:.3f}")


# ================================================================
# Helper: pick black or white text for contrast
# ================================================================
def _text_color(rgba):
    """Return 'white' or 'black' for best contrast against rgba background."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.55 else 'black'


# ================================================================
# PLOT 1: Sequential — circular chain
# ================================================================
print("\nGenerating sequential circular chain plot...")

cmap = plt.cm.RdYlGn
n_nodes = len(seq_ci)

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.axis('off')

radius = 4.2
node_r = 0.48
angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_nodes,
                     endpoint=False)
xs = radius * np.cos(angles)
ys = radius * np.sin(angles)

# Draw edges (arrows)
for i in range(n_nodes - 1):
    dx = xs[i + 1] - xs[i]
    dy = ys[i + 1] - ys[i]
    d = np.sqrt(dx ** 2 + dy ** 2)
    shrink = (node_r + 0.04) / d
    ax.annotate(
        '', xy=(xs[i + 1] - dx * shrink, ys[i + 1] - dy * shrink),
        xytext=(xs[i] + dx * shrink, ys[i] + dy * shrink),
        arrowprops=dict(arrowstyle='->', color='#555', lw=2.0,
                        connectionstyle='arc3,rad=0.08'))

# Draw nodes
for i, (x, y, ci_val) in enumerate(zip(xs, ys, seq_ci)):
    rgba = cmap(ci_val)
    tc = _text_color(rgba)
    circle = plt.Circle((x, y), node_r, fc=rgba, ec='k', lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, f'{ci_val:.0%}', ha='center', va='center', fontsize=14,
            fontweight='bold', color=tc, zorder=6)

    # Outer label: overlap count
    lx = (radius + node_r + 0.5) * np.cos(angles[i])
    ly = (radius + node_r + 0.5) * np.sin(angles[i])
    if i == 0:
        lbl = 'Init'
    else:
        lbl = f'ov={seq_overlaps[i]}'
    ax.text(lx, ly, lbl, ha='center', va='center', fontsize=11,
            fontweight='bold', color='#333')

ax.set_xlim(-radius - 1.8, radius + 1.8)
ax.set_ylim(-radius - 1.8, radius + 1.8)
ax.set_title('Sequential lstsq alignment — 14 patches\n'
             'Node color/label = column info preserved',
             fontsize=16, fontweight='bold', pad=18)

fig.tight_layout()
fig.savefig('plots/sequential_chain_14p.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved plots/sequential_chain_14p.png")

# ================================================================
# PLOT 2: Hierarchical — binary tree
# ================================================================
print("Generating hierarchical tree plot...")


def _tree_layout(nid, node_info, x_counter=None):
    """Compute (x, depth) for each node via in-order traversal."""
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
pos = {nid: (x * 1.8, y * 2.2) for nid, (x, y) in raw_pos.items()}

fig, ax = plt.subplots(figsize=(20, 10))
ax.axis('off')

node_r_h = 0.55

# Edges
for nid, nfo in node_info.items():
    if nfo['children'] is not None:
        px, py = pos[nid]
        for child_id in nfo['children']:
            cx, cy = pos[child_id]
            ax.plot([px, cx], [py, cy], color='#888', lw=2, zorder=1)

# Nodes
for nid in pos:
    x, y = pos[nid]
    nfo = node_info[nid]
    val = min(nfo['ci'], 1.0)
    rgba = cmap(val)
    tc = _text_color(rgba)

    circle = plt.Circle((x, y), node_r_h, fc=rgba, ec='k', lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, f'{val:.0%}', ha='center', va='center', fontsize=13,
            fontweight='bold', color=tc, zorder=6)

    if nfo['children'] is None:
        ax.text(x, y - node_r_h - 0.18, nfo['label'],
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='#333')
    else:
        ov_txt = f"ov={nfo.get('n_overlap', '?')}"
        ax.text(x, y + node_r_h + 0.15, ov_txt,
                ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='#555')

all_x = [x for x, _ in pos.values()]
all_y = [y for _, y in pos.values()]
ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
ax.set_ylim(min(all_y) - 1.2, max(all_y) + 1.5)
ax.set_aspect('equal')

ax.set_title('Hierarchical lstsq alignment — 14 patches\n'
             'Node color/label = column info preserved',
             fontsize=16, fontweight='bold', pad=18)

fig.tight_layout()
fig.savefig('plots/hierarchical_tree_14p.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved plots/hierarchical_tree_14p.png")

print("\nDone.")
