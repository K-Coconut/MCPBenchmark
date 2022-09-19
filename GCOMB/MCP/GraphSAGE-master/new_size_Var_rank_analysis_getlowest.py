import os
import json
import time
from networkx.readwrite import json_graph
import networkx as nx
from scipy import interpolate
import pickle
import sys

budget_list = [20, 50, 100, 150, 200]
dataset = sys.argv[1]
for num_k in budget_list:
    interpolator_file = 'interpolator/interpolate_budget_percentage_real_budget_{}{}.pkl'.format(dataset, num_k)

    start_time = time.time()
    time_log = open("interpolator/time_{}_{}.txt".format(dataset, num_k), 'w')

    size_var = [50, 80, 90, 99]
    x_axis = []
    y_axis = []

    prev_max = 0

    for size in size_var:
        graph = './real_data/{}/train{}/_bp-edges.txt'.format(dataset, size)
        sol = graph + ".greedy" + str(num_k)
        sup_gs_sol = graph + "_sup_GS_sol.txt"

        os.system("pwd")
        solution_file = open(sol, "r")
        solution_file.readline()
        optimal_nodes = solution_file.read().split()
        optimal_nodes = [int(x.replace('\n', '')) for x in optimal_nodes]

        G = nx.read_edgelist(graph, nodetype=int)
        degree_of_nodes = G.degree()

        degrees_array = sorted([x[1] for x in degree_of_nodes.items()], reverse=True)

        sorted_ids_deg_wt = sorted(range(len(degree_of_nodes)), key=lambda k: degree_of_nodes[k], reverse=True)
        min_score = 10000000
        max_rankn = -1000
        max_rank_id = -10000
        ranks_list = []

        for sol_id in optimal_nodes:
            res = len(degrees_array) - 1 - degrees_array[::-1].index(degree_of_nodes[sol_id])
            ranks_list.append(res)
            if (res > max_rankn):
                max_rankn = res

        print("budget", num_k, "max rank id", max_rank_id, "max rank ", max_rankn, "min score", min_score)

        len_G = len(G)
        print(" nodes in graph", len_G)

        x_axis.append(len_G / 2)

        prev_max = max(prev_max, int(max_rankn) / 2 + 1)

        y_axis.append(prev_max)

    print(x_axis)
    print(y_axis)
    f_interp = interpolate.interp1d(x_axis, y_axis, fill_value="extrapolate")
    file_dict = open(interpolator_file, 'wb')
    pickle.dump(f_interp, file_dict)
    end_time = time.time()
    print("{} Interpolator budget {} costs: {}".format(dataset, num_k, end_time - start_time))
    time_log.write(str(end_time - start_time))
