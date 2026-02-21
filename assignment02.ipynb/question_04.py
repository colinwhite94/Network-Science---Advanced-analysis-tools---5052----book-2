# run in terminal with correct environment:
# conda activate gtcnet5052
# export PATH="/Users/colinwhite/miniconda3/envs/gtcnet5052/bin:$PATH"
# python3 /Users/colinwhite/Desktop/cnet5052/assignments/assignment02.ipynb/question_04.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

# this script simulates a planted partition sbm runs louvain multiple 
# times builds a consensus matrix and compares partitions with nmi and ami

# part b, generate planted-partition SBM graph
p_in = 0.08 # within-block
p_out = 0.005 # between-block
sizes = [100, 200, 300, 400]  # N = 1000
num_blocks = len(sizes)

# build probability matrix
p_matrix = []
for i in range(num_blocks):
    row = []
    for j in range(num_blocks):
        row.append(p_in if i == j else p_out)
    p_matrix.append(row)


print("within-block probability (p_in):", p_in)
print("between-block probability (p_out):", p_out)
print(f"expected within-block edges per node: ~{p_in * 100:.1f}")
print(f"expected between-block edges per node: ~{p_out * 900:.1f}")

# build graph
G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
print(f"\nnodes: {G.number_of_nodes()}")
print(f"edges: {G.number_of_edges()}")

# build planted partition
planted_partition = {}
node = 0
# assign community labels based on block sizes
for block_id, size in enumerate(sizes):
    for _ in range(size):
        planted_partition[node] = block_id
        node += 1

print(f"\nplanted partition (first 10 entries): { {k: planted_partition[k] for k in range(10)} }")
print(f"community sizes: { {i: sizes[i] for i in range(num_blocks)} }")

# part c, run Louvain R=100 times
R = 100
partitions = []
modularities = []
num_communities = []

# run louvain multiple times to see variability in results
for r in range(R):
    # each run uses a different random seed to explore variability in results
    partition = community_louvain.best_partition(G, random_state=r)
    mod = community_louvain.modularity(partition, G)
    n_comm = len(set(partition.values()))
    partitions.append(partition)
    modularities.append(mod)
    num_communities.append(n_comm)
    if (r + 1) % 10 == 0:
        print(f"run {r+1}/100 — modularity: {mod:.4f}, communities: {n_comm}")

print(f"\nmean modularity: {np.mean(modularities):.4f} ± {np.std(modularities):.4f}")
print(f"mean communities: {np.mean(num_communities):.1f} ± {np.std(num_communities):.1f}")

# plot distributions of modularity and number of communities across runs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Louvain Community Detection — 100 Runs", fontsize=14, fontweight="bold")

# plot modularity distribution
mod_bins = 20 if len(set(modularities)) > 1 else 1
axes[0].hist(modularities, bins=mod_bins, color="steelblue", edgecolor="white")
axes[0].axvline(np.mean(modularities), color="red", linestyle="--",
                label=f"Mean={np.mean(modularities):.4f}")
axes[0].set_xlabel("Modularity", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].set_title("Distribution of Modularity Values", fontsize=12)
axes[0].legend()

# plot number of communities distribution
comm_bins = list(range(min(num_communities), max(num_communities) + 2)) \
    if len(set(num_communities)) > 1 else \
    [min(num_communities) - 0.5, min(num_communities) + 0.5]
axes[1].hist(num_communities, bins=comm_bins, color="steelblue",
             edgecolor="white", align="left")
axes[1].axvline(np.mean(num_communities), color="red", linestyle="--",
                label=f"Mean={np.mean(num_communities):.1f}")
axes[1].set_xlabel("Number of Communities", fontsize=12)
axes[1].set_ylabel("Count", fontsize=12)
axes[1].set_title("Distribution of Number of Communities", fontsize=12)
axes[1].legend()

# save plot
plt.tight_layout()
plt.savefig("question_04_c.png", dpi=200, bbox_inches="tight")
print("saved: question_04_c.png")
plt.close()

# part d, build and visualize consensus matrix

# builds consensus matrix C where C[i,j] 
# is the fraction of runs where nodes i and
#  j were co-assigned to the same community
nodes = list(G.nodes())
n = len(nodes)
node_to_idx = {node: i for i, node in enumerate(nodes)}

# compute consensus matrix C where 
# C[i, j] = fraction of runs where nodes 
# i and j are in the same community
C = np.zeros((n, n))
# iterate over all partitions and count co-assignments
for partition in partitions:
    # iterate over all pairs of nodes (i, j) 
    # and check if they are in the same community
    for i in range(n):
        for j in range(i, n):
            if partition[nodes[i]] == partition[nodes[j]]:
                C[i, j] += 1
                C[j, i] += 1

C /= R
# set diagonal to 1 (each node is always co-assigned with itself)
np.fill_diagonal(C, 1.0)

print(f"consensus matrix shape: {C.shape}")
print(f"mean off-diagonal value: {C[np.triu_indices(n, k=1)].mean():.4f}")

# sort nodes by planted community for better visualization
sorted_nodes = sorted(nodes, key=lambda x: planted_partition[x])
# build sorted consensus matrix
sorted_idx = [node_to_idx[nd] for nd in sorted_nodes]
# plot consensus heatmap
C_sorted = C[np.ix_(sorted_idx, sorted_idx)]

# plot consensus heatmap
fig, ax = plt.subplots(figsize=(8, 7))
# plot heatmap of consensus matrix
im = ax.imshow(C_sorted, cmap="viridis", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Fraction of runs co-assigned")

# add lines to show block boundaries
boundaries = [0, 100, 300, 600, 1000]
# add red dashed lines to indicate planted community boundaries
for b in boundaries[1:-1]:
    ax.axhline(b - 0.5, color="red", linewidth=0.8, linestyle="--")
    ax.axvline(b - 0.5, color="red", linewidth=0.8, linestyle="--")

# add labels and title
ax.set_title("Consensus Matrix C\n(nodes sorted by planted community)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Node index (sorted by community)", fontsize=11)
ax.set_ylabel("Node index (sorted by community)", fontsize=11)
plt.tight_layout()
plt.savefig("question_04_d.png", dpi=200, bbox_inches="tight")
print("saved: question_04_d.png")
plt.close()

# part e, build consensus graph and run community detection

# tau=0.5: selects edges where nodes were co-assigned in the majority
# of runs, separating strong within-block signal (C~1.0)
# from between-block noise (C~0.0)
tau = 0.5

# build consensus graph by thresholding consensus matrix
# add edge (i, j) if C[i, j] >= tau
# this creates a new graph where edges represent strong co-assignment
# we can then run community detection on this consensus graph 
# to see if it recovers the planted partition better than individual runs
G_cons = nx.Graph()
# add all nodes to consensus graph
G_cons.add_nodes_from(nodes)
for i in range(n):
    for j in range(i + 1, n):
        if C[i, j] >= tau:
            G_cons.add_edge(nodes[i], nodes[j], weight=C[i, j])

print(f"threshold tau: {tau}")
print(f"consensus graph — nodes: {G_cons.number_of_nodes()}, edges: {G_cons.number_of_edges()}")

# run community detection on consensus graph
# we can use the same louvain method, but now on the consensus graph
partition_cons = community_louvain.best_partition(G_cons, random_state=42, weight="weight")
mod_cons = community_louvain.modularity(partition_cons, G_cons, weight="weight")
num_comm_cons = len(set(partition_cons.values()))

print(f"consensus partition modularity: {mod_cons:.4f}")
print(f"consensus partition communities: {num_comm_cons}")
print(f"partition (first 10 entries): { {k: partition_cons[k] for k in list(partition_cons)[:10]} }")

# part f, compare consensus partition to planted partition with NMI and AMI
b_planted = [planted_partition[nd] for nd in nodes]
b_cons    = [partition_cons[nd] for nd in nodes]
b_single  = [partitions[0][nd] for nd in nodes]

# compute NMI and AMI for both single run and consensus partition
nmi_cons   = normalized_mutual_info_score(b_planted, b_cons)
nmi_single = normalized_mutual_info_score(b_planted, b_single)
ami_cons   = adjusted_mutual_info_score(b_planted, b_cons)
ami_single = adjusted_mutual_info_score(b_planted, b_single)

print(f"\nnmi  — single run: {nmi_single:.4f} | consensus: {nmi_cons:.4f}")
print(f"ami  — single run: {ami_single:.4f} | consensus: {ami_cons:.4f}")

# plot consensus vs planted partition
pos = nx.spring_layout(G, seed=42)
# plot consensus vs planted partition
cmap = plt.colormaps.get_cmap("tab10")
max_comm = max(partition_cons.values())

# plot consensus vs planted partition
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Consensus Partition vs Planted Partition", fontsize=14, fontweight="bold")

# plot planted partition with colors based on planted community
node_colors_planted = [cmap(planted_partition[nd] / 3) for nd in G.nodes()]
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.2, edge_color="gray", ax=axes[0])
nx.draw_networkx_nodes(G, pos, node_color=node_colors_planted, node_size=15, ax=axes[0])
axes[0].set_title("Planted Partition (ground truth)", fontsize=12)
axes[0].axis("off")

# plot consensus partition with colors based on consensus communities
node_colors_cons = [cmap(partition_cons[nd] / (max_comm if max_comm > 0 else 1)) for nd in G.nodes()]
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.2, edge_color="gray", ax=axes[1])
nx.draw_networkx_nodes(G, pos, node_color=node_colors_cons, node_size=15, ax=axes[1])
axes[1].set_title(f"Consensus Partition (τ={tau})\nNMI={nmi_cons:.4f}, Communities={num_comm_cons}",
                  fontsize=12)
axes[1].axis("off")

# save plot
plt.tight_layout()
plt.savefig("question_04_e.png", dpi=200, bbox_inches="tight")
print("saved: question_04_e.png")
plt.close()

