import os
import argparse

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True)
    args.add_argument("-m", "--weight_model", required=True)
    args.add_argument('--n-iter', dest='n_iter', type=int, default=3, help='repeated times of experiments')
    args.add_argument("-p", "--model_prefix", default="1_trainset_", help="the prefix of the model used to evaluate")
    args = args.parse_args()

    dataset = args.dataset
    weight_model = args.weight_model
    model_prefix = args.model_prefix
    n_iter = args.n_iter
    DATA_ROOT = os.path.join("result", dataset, weight_model, model_prefix)
    budget = [20, 50, 100, 150, 200]
    for k in budget:
        coverage, runtime, memory_usage = 0., 0., 0.
        if not os.path.exists(os.path.join(DATA_ROOT, f"seed{k}_n_0.txt")): continue
        for n in range(n_iter):
            coverage += float(open(os.path.join(DATA_ROOT, f"coverage_{k}_{n}.txt"), 'r').read())
            runtime += float(open(os.path.join(DATA_ROOT, f"time_budget{k}_n_{n}.txt"), 'r').read())
            memory_usage += float(open(os.path.join(DATA_ROOT, f"mem_budget{k}_n_{n}.txt"), 'r').read())
        with open(os.path.join(DATA_ROOT, f"budget{k}.txt"), 'w') as f:
            f.write(f"{coverage / n_iter} {runtime / n_iter} {memory_usage / n_iter}")
        print(f"{dataset} {weight_model}, k: {k}\tcoverage:{coverage / n_iter}\truntime:{runtime / n_iter}\tmemroy usage: {memory_usage / n_iter}")
