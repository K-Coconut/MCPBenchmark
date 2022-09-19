import os
import sys
import time

import numpy as np
from pathlib import Path

from networkx.algorithms import bipartite

import evaluate
from greedy_old import read_json_file, actual_greedy

budgets = np.arange(300, 5000, 100)

for dataset_ in [sys.argv[1]]:
    for size_var in [50, 80, 90, 99]:
        folder = '../real_data/'
        type = "train" + str(size_var)
        dataset = dataset_ + "/" + type + "/"
        input_file = folder + "/{}/_bp".format(dataset) + "-G.json"
        main_graph = read_json_file(input_file)
        set_nodes, element_nodes = bipartite.sets(main_graph)
        set_nodes = list(set_nodes)
        element_nodes = list(element_nodes)

        for budget in budgets:
            topKFile = folder + "{}/_bp".format(dataset) + "-G.json.greedy" + str(budget)
            if Path(topKFile).exists():
                print("File {} exists, skip".format(topKFile))
                continue

            reward_file = folder + "/{}/_bp".format(dataset) + "-G.json.greedy_reward" + str(budget)

            start_time = time.time()
            k = budget
            topk = actual_greedy(main_graph, set_nodes, budget)
            end_time = time.time()
            time_elapsed = end_time - start_time
            topKFile = open(topKFile, 'w')
            topKFile.write(str(len(topk)) + '\n')
            for node in topk:
                topKFile.write(str(node) + ' ')
            topKFile.close()
            print(reward_file)

            rewardFile = open(reward_file, 'w')
            rewardFile.write(str(evaluate.evaluate(main_graph.copy(), topk)))
            rewardFile.write('\n' + str(time_elapsed))
            rewardFile.close()
