import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import community as community_louvain  # pip install python-louvain

# this script simulates ba networks to estimate modularity 
# as a function of size and m parameter, then compares to an analytical prediction

# helper functions for modularity calculations
def get_modularity(G):
    #find maximum modularity using the Louvain algorithm (approximates simulated annealing)"""
    partition = community_louvain.best_partition(G)
    return community_louvain.modularity(partition, G)

def analytical_modularity(S, m, a=0.165):
    #equation 21 from the paper
    return (a + (1 - a) / m) * (1 - 2 / np.sqrt(S))

# set experiment parameters
m_values = [1, 2, 3, 4, 5]
S_values = [50, 100, 150, 200, 250, 300]
n_trials = 10  # average over multiple networks per (S, m) pair

# store average modularity values
results = {}  # (m, S) -> mean modularity

# run simulations across parameters
for m in m_values:
    results[m] = {}
    for S in tqdm(S_values, desc=f"m={m}"):
        modularities = []
        for _ in range(n_trials):
            G = nx.barabasi_albert_graph(S, m)
            mod = get_modularity(G)
            modularities.append(mod)
        results[m][S] = np.mean(modularities)

# configure plot styling
markers = {1: 'o', 2: 's', 3: 'D', 4: '^', 5: '<'}
fillstyles = {1: 'none', 2: 'full', 3: 'none', 4: 'full', 5: 'none'}
labels = {1: 'm = 1', 2: 'm = 2', 3: 'm = 3', 4: 'm = 4', 5: 'm = 5'}

fig, ax = plt.subplots(figsize=(6, 5))

# create smooth size values for analytical curve
S_smooth = np.linspace(50, 400, 300)

for m in m_values:
    # plot numerical results
    S_vals = list(results[m].keys())
    mod_vals = list(results[m].values())
    ax.plot(S_vals, mod_vals, 
            marker=markers[m], 
            fillstyle=fillstyles[m],
            linestyle='none',
            color='black',
            label=labels[m])
    
    # plot analytical prediction
    analytical = [analytical_modularity(S, m) for S in S_smooth]
    ax.plot(S_smooth, analytical, '-', color='black', linewidth=0.8)

ax.set_xlabel('Size, S')
ax.set_ylabel('Modularity')
ax.set_xlim(0, 400)
ax.set_ylim(0.2, 1.0)
ax.legend(title='Eq. (21)', loc='upper left')

ax.set_title('Modularity in scale-free networks')
plt.tight_layout()
plt.savefig("question_02.png", dpi=200, bbox_inches="tight")
plt.close()