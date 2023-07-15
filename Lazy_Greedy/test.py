import time
import os
import sys
import argparse
import tracemalloc

from alg import lazy_greedy

sys.path.append(os.path.join(".."))
from utils.Graph import Graph
from utils.mcp_evaluate import calculate_coverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset', required=True)
    parser.add_argument("-n", '--repeat_num', type=int, default=5)
    args = parser.parse_args()
    dataset = args.dataset
    tracemalloc.start()

    output_path = f"result/{dataset}"
    os.makedirs(output_path, exist_ok=True)
    inputFileName = os.path.join("..", "data", dataset, "edges.txt")
    budgets = [20, 50, 100, 150, 200]

    main_graph = Graph()
    main_graph.read_edges(inputFileName)
    for k in budgets:
        result_file = os.path.join(output_path, f"budget{k}.txt")
        total_runtime = 0.
        total_coverage = 0.
        for i in range(args.repeat_num):
            start_time = time.time()
            solution_set = lazy_greedy(main_graph, k)
            coverage = calculate_coverage(main_graph, solution_set)
            end_time = time.time()
            total_runtime += end_time - start_time
            total_coverage += coverage
        result_file = open(result_file, 'w')
        print("k: {}\treward: {}\ttime: {}".format(k, total_coverage / args.repeat_num, total_runtime / args.repeat_num))
        result_file.write(f"{total_coverage / args.repeat_num:.4f}\t{total_runtime / args.repeat_num:.4f}")
        current, peak = tracemalloc.get_traced_memory()
        print("Current memory usage is %.3f MB; Peak was %.3f MB" % (current / 10 ** 6, peak / 10 ** 6))
        f = open(os.path.join(output_path, "memory_budget%d.txt" % k), 'w')
        f.write("%.3f" % (peak / 10 ** 6))