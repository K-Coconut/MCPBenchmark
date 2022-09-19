import os
import networkx as nx
from scipy import interpolate
import pickle
import argparse

args = argparse.ArgumentParser()
args.add_argument('-d', '--dataset', required=True)
args.add_argument('-m', '--weight_model', required=True)
args.add_argument('-k', '--budget', type=int, required=True)
args = args.parse_args()

dataset = args.dataset
weight_model = args.weight_model
num_k = args.budget

output_dir = os.path.join("interpolator", weight_model)
os.makedirs(output_dir, exist_ok=True)

interpolate_output_file = os.path.join(output_dir, f"interpolate_budget_percentage_real_budget_{dataset}{num_k}.pkl")
epsilon = 0.5
size_var = [50, 80, 90, 99]
x_axis = []
y_axis = []
for size in size_var:
    data_dir = os.path.join("real_data", dataset, weight_model, f"train{size}")
    sol = os.path.join(data_dir, "multi_iter",
                       f"large_graph_ic_imm_sol_eps{epsilon}_num_k_{num_k}_iter_0_seeds_IC_{epsilon}.txt")
    optimal_nodes = open(sol, "r").read().split("\n")
    optimal_nodes = [int(x) for x in optimal_nodes if x]
    G = nx.read_edgelist(os.path.join(data_dir, "edges.txt"), nodetype=int, data=(('weight', float),))
    degree_of_nodes = G.degree()
    sorted_ids_deg_wt = sorted(range(len(degree_of_nodes)), key=lambda k: degree_of_nodes[k], reverse=True)

    min_score = 10000000
    max_rankn = -1000
    max_rank_id = -10000
    ranks_list = []

    for sol_id in optimal_nodes:
        ranks_list.append(sorted_ids_deg_wt.index(sol_id))
        if degree_of_nodes[sol_id] < min_score:
            min_score = degree_of_nodes[sol_id]
            max_rankn = max(max_rankn, min(loc for loc, val in enumerate(sorted_ids_deg_wt) if
                                           degree_of_nodes[val] == min_score))
            max_rank_id = sol_id

    sorted_ranks_list = sorted(ranks_list)
    second_last_rank = sorted_ranks_list[-2]
    print("budget", num_k, "max rank id", max_rank_id, "max rank ", max_rankn, "min score", min_score)

    len_G = len(G)
    x_axis.append(len_G)
    y_axis.append(max_rankn)

print(x_axis)
print(y_axis)
f_interp = interpolate.interp1d(x_axis, sorted(y_axis), fill_value="extrapolate")

file_dict = open(interpolate_output_file, 'wb')
pickle.dump(f_interp, file_dict)
