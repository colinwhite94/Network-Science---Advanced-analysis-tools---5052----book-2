# reminder to conda activate gtcnet5052
# export PATH="/Users/colinwhite/miniconda3/envs/gtcnet5052/bin:$PATH"
# python3 /Users/colinwhite/Desktop/cnet5052/assignments/assignment02.ipynb/question_03_a_e.py

import graph_tool.all as gt
import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter

#this script loads the dnc email edge graph
#runs louvain and sbm community detection
#randomizes degrees and plots size distributions
#plots all results and saves the figures

# load graph
G_nx = nx.Graph()
with open("email-dnc.edges", "r", encoding="utf-8-sig") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            u, v = parts[0], parts[1]
            G_nx.add_edge(u, v)

print(f"Nodes: {G_nx.number_of_nodes()}")
print(f"Edges: {G_nx.number_of_edges()}")

pos = nx.spring_layout(G_nx, seed=42)

# draw community plot and save figure
def draw_communities(G, pos, partition, title, filename):
    num_comm = len(set(partition.values()))
    max_comm = max(partition.values())
    cmap = plt.colormaps.get_cmap("tab20")
    node_colors = [cmap(partition[node] / (max_comm if max_comm > 0 else 1)) for node in G.nodes()]
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.3, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, alpha=0.9, ax=ax)
    handles = [plt.scatter([], [], color=cmap(i / (max_comm if max_comm > 0 else 1)),
               label=f"Community {i}", s=40) for i in range(num_comm)]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2, title="Community")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()

# run louvain on original graph
partition_louvain = community_louvain.best_partition(G_nx)
mod_louvain = community_louvain.modularity(partition_louvain, G_nx)
num_communities_louvain = len(set(partition_louvain.values()))
print(f"Modularity: {mod_louvain:.4f}")
print(f"Number of communities: {num_communities_louvain}")
print(f"Partition (first 10 entries): { {k: partition_louvain[k] for k in list(partition_louvain)[:10]} }")
draw_communities(G_nx, pos, partition_louvain,
    f"DNC Email Network — Louvain\nModularity={mod_louvain:.4f}, Communities={num_communities_louvain}",
    "question_03_a.png")

# run sbm on original graph
g = gt.Graph(directed=False)
node_list = list(G_nx.nodes())
node_to_idx = {node: i for i, node in enumerate(node_list)}
g.add_vertex(len(node_list))
for u, v in G_nx.edges():
    g.add_edge(node_to_idx[u], node_to_idx[v])

# apply sbm
# using graph-tool's built-in method to find the optimal partition
state = gt.minimize_blockmodel_dl(g)
blocks = state.get_blocks()
partition_gt = {node_list[i]: int(blocks[i]) for i in range(len(node_list))}
unique_blocks = sorted(set(partition_gt.values()))
remap = {old: new for new, old in enumerate(unique_blocks)}
partition_gt = {node: remap[comm] for node, comm in partition_gt.items()}
num_communities_gt = len(set(partition_gt.values()))
mod_gt = community_louvain.modularity(partition_gt, G_nx)
print(f"Modularity: {mod_gt:.4f}")
print(f"Number of communities: {num_communities_gt}")
print(f"Partition (first 10 entries): { {k: partition_gt[k] for k in list(partition_gt)[:10]} }")
draw_communities(G_nx, pos, partition_gt,
    f"DNC Email Network — SBM\nModularity={mod_gt:.4f}, Communities={num_communities_gt}",
    "question_03_b.png")

# randomize graph while preserving degree
print("\nc: Degree-Preserving Randomization")
G_rand = nx.configuration_model([d for n, d in G_nx.degree()])
G_rand = nx.Graph(G_rand)
G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
print(f"Randomized — Nodes: {G_rand.number_of_nodes()}, Edges: {G_rand.number_of_edges()}")
pos_rand = nx.spring_layout(G_rand, seed=42)

print("\nc1: louvain on randomized graph")
# run louvain on randomized graph
partition_louvain_rand = community_louvain.best_partition(G_rand)
mod_louvain_rand = community_louvain.modularity(partition_louvain_rand, G_rand)
num_communities_louvain_rand = len(set(partition_louvain_rand.values()))
print(f"Modularity: {mod_louvain_rand:.4f}")
print(f"Number of communities: {num_communities_louvain_rand}")
print(f"Partition (first 10 entries): { {k: partition_louvain_rand[k] for k in list(partition_louvain_rand)[:10]} }")
draw_communities(G_rand, pos_rand, partition_louvain_rand,
    f"Randomized Graph — Louvain\nModularity={mod_louvain_rand:.4f}, Communities={num_communities_louvain_rand}",
    "question_03_c_louvain.png")

print("\nc2: SBM on randomized graph")
# run sbm on randomized graph
g_rand = gt.Graph(directed=False)
node_list_rand = list(G_rand.nodes())
node_to_idx_rand = {node: i for i, node in enumerate(node_list_rand)}
g_rand.add_vertex(len(node_list_rand))
for u, v in G_rand.edges():
    g_rand.add_edge(node_to_idx_rand[u], node_to_idx_rand[v])

print("applying sbm on randomized graph")
state_rand = gt.minimize_blockmodel_dl(g_rand)
blocks_rand = state_rand.get_blocks()
# get partition and remap to consecutive ids
partition_gt_rand = {node_list_rand[i]: int(blocks_rand[i]) for i in range(len(node_list_rand))}
# remap to consecutive ids
unique_blocks_rand = sorted(set(partition_gt_rand.values()))
# remap to consecutive ids
remap_rand = {old: new for new, old in enumerate(unique_blocks_rand)}
# remap partition to consecutive ids
partition_gt_rand = {node: remap_rand[comm] for node, comm in partition_gt_rand.items()}
# compute modularity and number of communities
num_communities_gt_rand = len(set(partition_gt_rand.values()))
# compute modularity and number of communities
mod_gt_rand = community_louvain.modularity(partition_gt_rand, G_rand)
print(f"Modularity: {mod_gt_rand:.4f}")
print(f"Number of communities: {num_communities_gt_rand}")
print(f"Partition (first 10 entries): { {k: partition_gt_rand[k] for k in list(partition_gt_rand)[:10]} }")
draw_communities(G_rand, pos_rand, partition_gt_rand,
    f"Randomized Graph — SBM\nModularity={mod_gt_rand:.4f}, Communities={num_communities_gt_rand}",
    "question_03_c_sbm.png")

# compute community size distributions
def community_sizes(partition):
    counts = Counter(partition.values())
    return sorted(counts.values(), reverse=True)

# compute community size distributions for all partitions
sizes_louvain = community_sizes(partition_louvain)
sizes_gt = community_sizes(partition_gt)
sizes_louvain_rand = community_sizes(partition_louvain_rand)
sizes_gt_rand = community_sizes(partition_gt_rand)

# finding shared axis limits
all_sizes = sizes_louvain + sizes_gt + sizes_louvain_rand + sizes_gt_rand
all_counts = [len(sizes_louvain), len(sizes_gt), len(sizes_louvain_rand), len(sizes_gt_rand)]
x_max = max(all_counts) + 1
y_max = max(all_sizes) * 3
y_min = 0.8

# plot size distributions with log scale and annotations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Community Size Distributions", fontsize=16, fontweight="bold")

# plot size distributions with log scale and annotations
datasets = [
    (sizes_louvain,f"Original — Louvain (Q={mod_louvain:.3f})",axes[0, 0]),
    (sizes_gt, f"Original — SBM (Q={mod_gt:.3f})", axes[0, 1]),
    (sizes_louvain_rand, f"Randomized — Louvain (Q={mod_louvain_rand:.3f})", axes[1, 0]),
    (sizes_gt_rand, f"Randomized — SBM (Q={mod_gt_rand:.3f})",axes[1, 1]),
]

# plot each distribution with log scale, annotations, and shared axis limits
for sizes, title, ax in datasets:
    ranks = range(1, len(sizes) + 1)
    ax.bar(ranks, sizes, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Community Rank (by size)", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(0.5, x_max)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)
    for rank, size in zip(ranks, sizes):
        ax.text(rank, size * 1.05, str(size), ha="center", va="bottom", fontsize=7)
    ax.text(0.97, 0.97,
            f"n={len(sizes)} communities\nmax={max(sizes)}, min={min(sizes)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

# adjust layout and save figure
plt.tight_layout()
plt.savefig("question_03_size_distributions.png", dpi=200, bbox_inches="tight")
print("Saved: question_03_size_distributions.png")
plt.close()