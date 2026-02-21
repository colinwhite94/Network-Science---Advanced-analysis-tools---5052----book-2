import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

#this script loads two mystery graphs
#computes ba stats and estimates m for each graph
#generates ba ensembles and compares metrics
#plots distribution panels and saves the figure

# load the two mystery graphs
G1 = nx.read_adjlist("mystery_edges1.txt")
G2 = nx.read_adjlist("mystery_edges2.txt")

# compute basic stats and estimate m for ba model
print("g1 stats:")
print(f"  nodes: {G1.number_of_nodes()}")
print(f"  edges: {G1.number_of_edges()}")
print(f"  edges/nodes ratio: {G1.number_of_edges() / G1.number_of_nodes():.2f}")
m1 = round(G1.number_of_edges() / G1.number_of_nodes())
print(f"  best m (mle estimate): {m1}")

print("\ng2 stats:")
print(f"  nodes: {G2.number_of_nodes()}")
print(f"  edges: {G2.number_of_edges()}")
print(f"  edges/nodes ratio: {G2.number_of_edges() / G2.number_of_nodes():.2f}")
m2 = round(G2.number_of_edges() / G2.number_of_nodes())
print(f"  best m (mle estimate): {m2}")

# generate ba graph ensembles for each mystery graph
n1 = G1.number_of_nodes()
n2 = G2.number_of_nodes()

ref1 = [nx.barabasi_albert_graph(n1, m1) for _ in tqdm(range(100), desc="Generating G1 ensemble")]
ref2 = [nx.barabasi_albert_graph(n2, m2) for _ in tqdm(range(100), desc="Generating G2 ensemble")]

print(f"\ngenerated 100 ba graphs for g1 (n={n1}, m={m1})")
print(f"generated 100 ba graphs for g2 (n={n2}, m={m2})")

# compute metrics for every ensemble graph
rows = []
for G in tqdm(ref1, desc="G1 ensemble"):
    t = nx.transitivity(G)
    a = nx.average_shortest_path_length(G)
    d = nx.degree_assortativity_coefficient(G)
    rows.append((t, a, d, "reference 1"))

for G in tqdm(ref2, desc="G2 ensemble"):
    t = nx.transitivity(G)
    a = nx.average_shortest_path_length(G)
    d = nx.degree_assortativity_coefficient(G)
    rows.append((t, a, d, "reference 2"))

# store all ensemble metrics in a table
df = pd.DataFrame(rows, columns=["Transitivity", "Avg. Shortest Path Length", "Assortativity", "Reference"])

# compute the same metrics for the mystery graphs
g1_transitivity = nx.transitivity(G1)
g1_avg_path = nx.average_shortest_path_length(G1)
g1_assortativity = nx.degree_assortativity_coefficient(G1)

g2_transitivity = nx.transitivity(G2)
g2_avg_path = nx.average_shortest_path_length(G2)
g2_assortativity = nx.degree_assortativity_coefficient(G2)

print(f"\ng1 - transitivity: {g1_transitivity:.4f}, avg path length: {g1_avg_path:.4f}, assortativity: {g1_assortativity:.4f}")
print(f"g2 - transitivity: {g2_transitivity:.4f}, avg path length: {g2_avg_path:.4f}, assortativity: {g2_assortativity:.4f}")

# plot distributions and add reference lines
df2 = df.melt(id_vars=["Reference"], var_name="Metric")

g = sns.FacetGrid(df2, col="Metric", row="Reference", sharex=False, sharey=False, despine=False, margin_titles=True)
g.map(sns.kdeplot, "value")

# red reference lines for mystery graphs
g.axes[0, 0].axvline(x=g1_transitivity, color='r', label='Mystery Graph')
g.axes[0, 1].axvline(x=g1_avg_path, color='r')
g.axes[0, 2].axvline(x=g1_assortativity, color='r')

g.axes[1, 0].axvline(x=g2_transitivity, color='r')
g.axes[1, 1].axvline(x=g2_avg_path, color='r')
g.axes[1, 2].axvline(x=g2_assortativity, color='r')

g.axes[0, 0].legend()
g.figure.suptitle("BA Ensemble vs Mystery Graphs", y=1.02)
plt.tight_layout()
plt.savefig("question_01.png", dpi=200, bbox_inches="tight")
plt.close()