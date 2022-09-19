import networkx as nx
import json

import sys
import os


def gen_setcover_inst(G):
    constant_add = len(G)
    print(" const add ", constant_add)
    node_list_A = list(set(G.nodes()))
    node_list_B = [x + constant_add for x in list(set(G.nodes()))]

    g = nx.Graph()
    g.add_nodes_from(node_list_A, bipartite=0)
    g.add_nodes_from(node_list_B, bipartite=1)

    for (v1, v2) in G.edges():
        if v1 == v2:
            continue
        g.add_edge(v1, v2 + constant_add)
        g.add_edge(v1 + constant_add, v2)

    return g


dataset = sys.argv[1]

train_or_test = "test"
# test data
DATA_ROOT = f"../../../../data/{dataset}/"
# training data
# DATA_ROOT = f"{dataset}/train/"

input_file_path = f"{DATA_ROOT}/edges.txt"
output_dir = f"{dataset}/{train_or_test}"
os.makedirs(output_dir, exist_ok=True)
graph_name = f"{output_dir}/_bp"
G = nx.read_edgelist(input_file_path, nodetype=int)
node_label = {nodeId: index for nodeId, index in zip(G.nodes(), range(G.number_of_nodes()))}
G = nx.relabel_nodes(G, node_label)
G = gen_setcover_inst(G)

total_nodes = len(G)
print(" total nodes ", total_nodes)
features = []

with open(graph_name + "-edges.txt", 'w') as f:
    for u in range(len(G)):
        for v in G.edge[u]:
            f.write(f"{u} {v}\n")

feats_file_name = graph_name + "-feats.npy"
json_id_map_name = graph_name + "-id_map.json"
class_map_file = graph_name + "-class_map.json"

id_map = {}
for node in range(0, total_nodes):
    id_map[str(node)] = node

iddata = json.dumps(id_map)
f2 = open(json_id_map_name, 'w')
f2.write(iddata)
f2.close()

class_map = {}
for node in range(0, total_nodes):
    class_map[str(node)] = [0]

classdata = json.dumps(class_map)
f2 = open(class_map_file, 'w')
f2.write(classdata)
f2.close()

with open(f"{output_dir}/attribute.txt", 'w') as f:
    f.write(f"n={len(G)}\nm={G.number_of_edges()}")
