import argparse
import random
import os
import networkx as nx
import tqdm
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
opt = args.parse_args()

SEED = 0
random.seed(SEED)
point_proportion = 20
dataset = opt.dataset

soln_budget = 100

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..",  "data", dataset)
output_dir = os.path.join(os.path.dirname(__file__), "data", dataset, "train")
os.makedirs(output_dir, exist_ok=True)
sub_graph_edge_file_path = os.path.join(output_dir, "edges.txt")
full_graph_edge_file_path = DATA_ROOT + "/edges.txt"
f_full_graph_edge = open(full_graph_edge_file_path, 'r')
f_sub_graph_edge = open(sub_graph_edge_file_path, 'w')
f_attribute = open(os.path.join(output_dir, "attribute.txt"), 'w')

node_dic = {}
sep = '\t'
counter = 0

# Split on edges rather than nodes.
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
        counter += 1
        if n1 not in node_dic:
            node_dic[n1] = len(node_dic)
        if n2 not in node_dic:
            node_dic[n2] = len(node_dic)
        u = node_dic[n1]
        v = node_dic[n2]
        f_sub_graph_edge.write(str(u) + ' ' + str(v) + '\n')

f_sub_graph_edge.close()
f_attribute.write(f"n={len(node_dic)}\nm={counter}")
