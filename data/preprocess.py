from operator import le
import networkx as nx
import numpy as np
import random
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("-m", '--weight_model', default='WC', help='weight model')
parser = args.parse_args()

SEED = 0
random.seed(SEED)
dataset = parser.dataset + "/"
print("dataset", dataset)
weight_model = parser.weight_model

input_file_name = os.path.join(dataset, "edges.txt")
folder = os.path.join(dataset, "IM", weight_model)
edges_file_path = os.path.join(folder, "edges.txt")
os.makedirs(folder, exist_ok=True)

G = nx.DiGraph(name=dataset)

print("input file of edges ", input_file_name)
file = open(input_file_name, 'r')
edges_file = open(edges_file_path, 'w')

node_dic = {}
while True:
    line = file.readline()
    if line.startswith("#"):
        continue
    if len(line) < 2:
        break
    edge = line.strip().split()
    source = int(edge[0].rstrip('\x00'))
    target = int(edge[1].rstrip('\x00'))

    if source not in node_dic:
        node_dic[source] = len(node_dic)

    if target not in node_dic:
        node_dic[target] = len(node_dic)

    G.add_edge(node_dic[source], node_dic[target])
    if weight_model == 'LND':
        weight = float(edge[2].rstrip('\x00'))
        G[source][target]['weight'] = weight

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
    edges_file.write(str(u) + ' ' + str(v) + ' ' + str(weight) + '\n')

edges_file.close()

attribute_file_name = os.path.join(folder, 'attribute.txt')
attribute_file = open(attribute_file_name, 'w')
attribute_file.write(f'n={G.number_of_nodes()}\nm={G.number_of_edges()}')
