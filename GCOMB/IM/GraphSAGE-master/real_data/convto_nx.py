import networkx as nx
import json
import numpy as np
import random
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("-t", '--mode', required=True, help='train, test or validation')
args.add_argument("-m", '--weight_model', default='WC', help='weight model')
args = args.parse_args()

SEED = 0
random.seed(SEED)
dataset = args.dataset
DATA_DIR = os.path.join("..", "..", "..", "..", "data", dataset)
weight_model = args.weight_model
mode = args.mode
OUTPUT_DIR = os.path.join(dataset, weight_model, mode)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NOTE: Test phase uses all the edges. Edges in train dataset just for interpolator, which has no impact on test phase.
input_file_path = os.path.join(DATA_DIR, "edges.txt")
output_edges_file_path = os.path.join(OUTPUT_DIR, "edges.txt")
input_file = open(input_file_path, 'r')
edges_file = open(output_edges_file_path, 'w')

G = nx.DiGraph(name=dataset)

node_dic = {}
train_size = 15  # the paper indicates that 5% provides the same result as the 25% does, we set 10 here.
validation_size = 15
test_size = 100  #
while True:
    line = input_file.readline()
    if line.startswith("#"):
        continue
    if len(line) < 2:
        break

    if mode == 'train':
        if random.randint(0, 100) > train_size:
            continue
    elif mode == 'validation':
        seed = random.randint(0, 100)
        if seed < train_size or seed > train_size + validation_size:
            continue
    elif mode == 'test':
        # To compare with other methods conveniently, we use entire graph to test, even though it might help GCOMB perform better.
        pass
    else:
        raise Exception("wrong mode")

    edges = line.strip()
    edge = edges.split()
    source = int(edge[0])
    target = int(edge[1])

    if source not in node_dic:
        node_dic[source] = len(node_dic)

    if target not in node_dic:
        node_dic[target] = len(node_dic)

    G.add_edge(node_dic[source], node_dic[target])
    if weight_model == 'LND':
        weight = float(edge[2])
        G[source][target]['weight'] = weight

for u in range(len(node_dic)):
    for v in G.edge[u]:
        if weight_model == 'TV':
            weight = random.choice([0.1, 0.01, 0.001])
        elif weight_model == 'WC':
            weight = 1 / G.in_degree(v)
        elif weight_model == 'CONST':
            weight = 0.1
        elif weight_model == 'LND':
            weight = G[u][v]['weight']
        else:
            assert False, "Wrong model"
        G[u][v]['weight'] = weight
        edges_file.write(f"{u} {v} {weight}\n")
edges_file.close()
print("total nodes", len(G))

features = []
sum_degree = 0
for node_i in G.nodes():
    features.append([])
    sum_degree = sum_degree + G.degree(node_i)

for node_i in G.nodes():
    norm_value = G.degree(node_i) * 1.0 / sum_degree
    features[node_i].append(norm_value)

# feats file
graph_name = os.path.join(OUTPUT_DIR, 'large_graph')
print("dumping to large_graph-feats.npy")
np.save(graph_name + "-feats.npy", features)

class_map = {}
for node in range(0, len(G)):
    class_map[node] = [0]
with open(graph_name + "-class_map.json", 'w') as f:
    classdata = json.dumps(class_map)
    f.write(classdata)

# attribute file
with open(os.path.join(OUTPUT_DIR, 'attribute.txt'), 'w') as f:
    f.write(f"n={G.number_of_nodes()}\nm={G.number_of_edges()}")
