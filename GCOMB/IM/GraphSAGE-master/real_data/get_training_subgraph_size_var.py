import os
import random
import argparse
import networkx as nx

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("-m", '--weight_model', default='WC', help='weight model')
args = args.parse_args()

dataset = args.dataset
weight_model = args.weight_model
DATA_ROOT = os.path.join(dataset, weight_model)
full_graph_edge_file_path = os.path.join(DATA_ROOT, "train", "edges.txt")

SEED = 0
random.seed(SEED)

for size_var in [50, 80, 90, 99]:
    output_dir = os.path.join(DATA_ROOT, f"train{size_var}")
    train_sub_graph_edge_file_path = os.path.join(output_dir, "edges.txt")
    attribute_file_path = os.path.join(output_dir, "attribute.txt")
    os.makedirs(output_dir, exist_ok=True)

    f_full_graph_edge = open(full_graph_edge_file_path, 'r')
    f_train_sub_graph_edge = open(train_sub_graph_edge_file_path, 'w')
    print(train_sub_graph_edge_file_path)

    G = nx.DiGraph(name=dataset)
    node_dict = {}
    while True:
        line = f_full_graph_edge.readline()
        if line.startswith('#'):
            continue
        if not line:
            break
        random_int = random.randint(0, 100)
        if random_int < size_var:
            n1, n2, wt = line.replace('\n', '').split(' ')
            if n1 not in node_dict:
                node_dict[n1] = len(node_dict)
            if n2 not in node_dict:
                node_dict[n2] = len(node_dict)
            G.add_edge(node_dict[n1], node_dict[n2])
            if weight_model == 'LND':
                weight = float(wt)
                G[node_dict[n1]][node_dict[n2]]['weight'] = weight

    for u in range(len(node_dict)):
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
            f_train_sub_graph_edge.write(f"{u} {v} {weight}\n")

    f_train_sub_graph_edge.close()
    attribute_file = open(attribute_file_path, 'w')
    attribute_file.write(f'n={len(node_dict)}\nm={G.number_of_edges()}')
