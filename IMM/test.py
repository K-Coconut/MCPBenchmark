import os
import random
import pathlib
import argparse

args = argparse.ArgumentParser()
args.add_argument('-m', '--weight_model', required=True)
args.add_argument('-p', '--path', default='../data/', help='input path')
args.add_argument('-d', '--dataset', required=True)
args.add_argument('-n', '--n_iter', default=3, help='number of iterations')
args.add_argument('--epsilon', default=0.5, type=float,
                  help='epsilon, default=0.5')
args = args.parse_args()

weight_model = args.weight_model
dataset = args.dataset
num_of_iterations = int(args.n_iter)
base_path = os.path.join(args.path, dataset, "IM", weight_model)
graph_path = os.path.join(base_path, "edges.txt")
if not os.path.exists(graph_path):
    raise Exception(f"Please preprocess to generate {graph_path}")

epsilon = args.epsilon
budgets = [10, 20, 50, 100, 150, 200]
output_folder = os.path.join("result", dataset, weight_model, "multi_iter")
os.makedirs(output_folder, exist_ok=True)

for k in budgets:
    for n in range(0, num_of_iterations):
        seed_random = random.randint(1, 100000000)
        output_prefix = os.path.join(output_folder, f"budget{k}_iter_{n}")
        seed_file = output_prefix + f"_seeds_IC_{epsilon}.txt"
        if pathlib.Path(seed_file).exists():
            print(f"File {seed_file} exists, continue")
            continue
        command = f"time ./imm_discrete -dataset {graph_path} -k {k} -model IC -epsilon {epsilon} -output {output_prefix} -seed_random {seed_random}"
        print(command)
        os.system(command)
