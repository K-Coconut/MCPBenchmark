import os
import argparse

args = argparse.ArgumentParser()
args.add_argument('-m', '--weight_model', default='TV')
args.add_argument('-a', '--algorithm', default='DDiscount',
                  help='select algorithm, DDiscount or SingleDiscount')
args.add_argument('-p', '--path', default='../data/', help='input path')
args.add_argument('-d', '--dataset', required=True)
args = args.parse_args()

weight_model = args.weight_model
dataset = args.dataset
algorithm = args.algorithm
graph_folder = os.path.join(args.path, dataset, "IM", weight_model) + "/"
graph_path = os.path.join(graph_folder, "edges.txt")
if not os.path.exists(graph_path):
    raise Exception(f"Please preprocess to generate {graph_path}")

budgets = [10, 20, 50, 100, 150, 200]
output_folder = os.path.join("result", dataset, weight_model)
os.makedirs(output_folder, exist_ok=True)

for k in budgets:
    output_prefix = os.path.join(output_folder, f"{algorithm}_budget{k}")
    command = f"time ./discount -graph_folder {graph_folder} -algorithm {algorithm} -k {k} -output_prefix {output_prefix}"
    print(command)
    os.system(command)
