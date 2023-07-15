import argparse
import random
import os
import networkx as nx
import tqdm

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("-m", '--weight_model', default='WC', help='weight model')
args.add_argument("-t", '--mode', default='train', help='train or test')
args.add_argument('--point_proportion', default=20, type=int, help="point proportion that training dataset takes up")
opt = args.parse_args()

dataset = opt.dataset
weight_model = opt.weight_model
assert opt.weight_model in ['TV', 'WC', 'CONST', 'LND']
assert opt.mode in ['train', 'test']

point_proportion = opt.point_proportion
default_point_proportion = 20
if point_proportion != default_point_proportion:
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", dataset, weight_model, "train_size_exp", f"train_size_{point_proportion}", opt.mode)
else:
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", dataset, weight_model, opt.mode)
os.makedirs(output_dir, exist_ok=True)

if opt.mode == 'test':
    point_proportion = 100 - point_proportion

SEED = 0
random.seed(SEED)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", dataset, "IM", weight_model)
full_graph_edge_file_path = DATA_ROOT + "/edges.txt"
sub_graph_edge_file_path = os.path.join(output_dir, "edges.txt")


f_full_graph_edge = open(full_graph_edge_file_path, 'r')
f_sub_graph_edge = open(sub_graph_edge_file_path, 'w')
f_attribute = open(os.path.join(output_dir, "attribute.txt"), 'w')


node_dic = {}
sep = '\t'
counter = 0

# Split on edges rather than nodes.
G = nx.DiGraph(name=dataset)
for line in tqdm.tqdm(f_full_graph_edge):
    if line[0] == "#":
        continue

    edge = line.replace('\n', '').split(sep)
    if len(edge) == 1:
        sep = ' '
        edge = line.replace('\n', '').split(sep)
    n1, n2 = edge[0], edge[1]
    random_int = random.randint(0, 100)
    if random_int < point_proportion:
        if n1 not in node_dic:
            node_dic[n1] = len(node_dic)
        if n2 not in node_dic:
            node_dic[n2] = len(node_dic)
            
        G.add_edge(node_dic[n1], node_dic[n2])
        if weight_model == 'LND':
            weight = float(edge[2].rstrip('\x00'))
            G[node_dic[n1]][node_dic[n2]]['weight'] = weight

for edge in G.edges():
    u, v = edge[0], edge[1]
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
    f_sub_graph_edge.write(str(u) + ' ' + str(v) + ' ' + str(weight) + '\n')


f_sub_graph_edge.close()
f_attribute.write(f"n={G.number_of_nodes()}\nm={G.number_of_edges()}")
