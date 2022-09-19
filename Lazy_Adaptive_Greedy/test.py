import time
import os
import sys
import argparse

from alg import lazy_adaptive_greedy

sys.path.append(os.path.join(".."))
from utils.Graph import Graph
from utils.mcp_evaluate import calculate_coverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset', required=True)
    parser.add_argument("-n", dest='N', type=int, default=10)
    args = parser.parse_args()
    dataset = args.dataset

    output_path = f"result/{dataset}"
    os.makedirs(output_path, exist_ok=True)
    inputFileName = os.path.join("..", "data", dataset, "edges.txt")
    budgets = [20, 50, 100, 150, 200]

    main_graph = Graph()
    main_graph.read_edges(inputFileName)
    for k in budgets:
        result_file = os.path.join(output_path, f"budget{k}.txt")
        start_time = time.time()
        solution_set = lazy_adaptive_greedy(main_graph, k, args.N)
        coverage = calculate_coverage(main_graph, solution_set)
        end_time = time.time()
        result_file = open(result_file, 'w')
        runtime = end_time - start_time
        print("k: {}\treward: {}\ttime: {}".format(k, coverage, runtime))
        result_file.write(f"{coverage}\t{runtime}")
