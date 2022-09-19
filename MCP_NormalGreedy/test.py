import argparse
import os
import sys
import time

from alg import greedy

sys.path.append(os.path.join(".."))
from utils.Graph import Graph
from utils.mcp_evaluate import calculate_coverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset', required=True)
    parser.add_argument("-k", "--budget", type=int, default=20)
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
        if os.path.exists(result_file):
            print("File {} exists, continue".format(result_file))
            continue
        start_time = time.time()
        solution_set = greedy(main_graph, k)
        coverage = calculate_coverage(main_graph, solution_set)
        end_time = time.time()
        runtime = end_time - start_time
        print("k: {}\treward: {}\ttime: {}".format(k, coverage, runtime))
        result_file = open(result_file, 'w')
        result_file.write(f"{coverage}\t{runtime}")
