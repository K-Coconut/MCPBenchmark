import os
import argparse

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True)
    args.add_argument("-m", "--weight_model", required=True)
    args = args.parse_args()

    dataset = args.dataset
    weight_model = args.weight_model
    DATA_ROOT = os.path.join("result", dataset, weight_model)
    budget = [20, 50, 100, 150, 200]
    for k in budget:
        coverage = float(open(os.path.join(DATA_ROOT, f"coverage_{k}.txt"), 'r').read())
        runtime = float(open(os.path.join(DATA_ROOT, f"time_budget{k}.txt"), 'r').read())
        with open(os.path.join(DATA_ROOT, f"budget{k}.txt"), 'w') as f:
            f.write(f"{str(coverage)} {str(runtime)}")
        print(f"{dataset} {weight_model}, k: {k}\tcoverage:{coverage}\truntime:{runtime}")
