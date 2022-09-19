import os
import random
import argparse

args = argparse.ArgumentParser()
args.add_argument('-d', '--dataset', required=True)
args.add_argument('-m', '--weight_model', required=True)
args.add_argument('-n', '--num', default=3, type=int, help='number of iterations')
args.add_argument('-s', '--size_var', default='', help='size of train set')
args = args.parse_args()

dataset = args.dataset
weight_model = args.weight_model
num_iter = args.num
size_var = args.size_var

IMM_PROGRAM = os.path.join("..", "..", "IMM", "imm_discrete")
budgets = [10, 20, 50, 100, 150, 200]
epsilon = 0.5

graph_path_dir = os.path.join("GraphSAGE-master", "real_data", dataset, weight_model, f"train{size_var}")
graph_path = os.path.join(graph_path_dir, "edges.txt")
output_path = os.path.join(graph_path_dir, "multi_iter")
os.makedirs(output_path, exist_ok=True)

for k in budgets:
    print(" GENERATING IMM REG ")
    for n in range(0, num_iter):
        print(" imm iteratin #", n)
        seed_random = random.randint(1, 100000000)
        output_prefix = os.path.join(output_path, f"large_graph_ic_imm_sol_eps{epsilon}_num_k_{k}_iter_{n}")
        command = f"time {IMM_PROGRAM} -dataset {graph_path} -k {k} -model IC -epsilon {epsilon} -output {output_prefix} -seed_random {seed_random} -training_for_gain 1"
        print("command imm ", command)
        os.system(command)
