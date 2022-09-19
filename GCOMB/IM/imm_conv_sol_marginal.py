import os
import json
import networkx as nx
import argparse

args = argparse.ArgumentParser()
args.add_argument('-d', '--dataset', required=True)
args.add_argument('-m', '--weight_model', required=True)
args.add_argument('-n', '--num', default=3, type=int, help='number of iterations')
args = args.parse_args()

dataset = args.dataset
weight_model = args.weight_model
num_iter = args.num

graph_path_dir = os.path.join("GraphSAGE-master", "real_data", dataset, weight_model, f"train")
dataset_path = os.path.join(graph_path_dir, "edges.txt")
G = nx.read_edgelist(dataset_path, nodetype=int, data=(('weight', float),))
total_nodes = len(G)
epsilon = 0.5
# set num_k as 50, which means we use num_k_50 class label
NUM_K = 50

dict_marginal_gain = {}

for n in range(0, num_iter):
    solution_file_name = os.path.join(graph_path_dir, "multi_iter",
                                      f"large_graph_ic_imm_sol_eps{epsilon}_num_k_{NUM_K}_iter_{n}_dict_node_gain.txt")

    for line in open(solution_file_name, "r"):
        node, gain = line.strip().split(" ")
        node, gain = int(node), int(gain)
        if node not in dict_marginal_gain:
            dict_marginal_gain[node] = 0

        dict_marginal_gain[node] += gain

sum_of_marginal_gains = 0
for node, marg_gain in dict_marginal_gain.items():
    sum_of_marginal_gains += marg_gain

dict_marginal_gain_normalized = {}
for node, marg_gain in dict_marginal_gain.items():
    dict_marginal_gain_normalized[node] = [marg_gain * 1.0 / sum_of_marginal_gains]

for node in range(0, total_nodes):
    if node not in dict_marginal_gain_normalized:
        dict_marginal_gain_normalized[node] = [0]

print("writing to large_graph-class_map.json")
class_map_file = os.path.join(graph_path_dir, "large_graph-class_map.json")
class_map = {}
with open(class_map_file, 'w') as f:
    classdata = json.dumps(dict_marginal_gain_normalized)
    f.write(classdata)
