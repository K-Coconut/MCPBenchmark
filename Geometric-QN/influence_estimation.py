import os
import argparse

INFLUENCE_EVALUATOR = '../IMEvaluation/evaluate'
DATA_DIR = '../data'


def evaluate(dataset, weight_model, model_prefix, budgets, size=1e6, n_iter=3):
    graph_file = os.path.join(DATA_DIR, dataset, "IM",
                              weight_model, 'edges.txt')
    klist = ",".join([str(i) for i in budgets]) + ","

    for i in range(n_iter):
        seed_file = os.path.join("result", dataset, weight_model,
                                 model_prefix, f'seed%d_n_{i}.txt')
        print(seed_file)
        output_file = os.path.join(
            "result", dataset, weight_model, model_prefix, f'coverage_%d_{i}.txt')
        command = f"{INFLUENCE_EVALUATOR} -seedFile {seed_file} -output {output_file} -graphFile {graph_file} -klist {klist} -size {size}"
        print(command)
        os.system(command)


def evaluate_with_epoch(dataset, weight_model, model_prefix, budgets, size=1e6, n_iter=20):
    graph_file = os.path.join(DATA_DIR, dataset, "IM",
                              weight_model, 'edges.txt')
    klist = ",".join([str(i) for i in budgets]) + ","
    file_dir = os.path.join(
        "result", dataset, weight_model, model_prefix, "epoch")
    epochs = [i for i in os.listdir(file_dir) if i.isnumeric()]
    for epoch in epochs:
        for n in range(n_iter):
            seed_file = os.path.join(
                file_dir, epoch, f'seed%d_n_iter_{n}.txt')
            output_file = os.path.join(
                file_dir, epoch, f'coverage_%d_n_iter_{n}.txt')
            command = f"{INFLUENCE_EVALUATOR} -seedFile {seed_file} -output {output_file} -graphFile {graph_file} -klist {klist} -size {size}"
            print(command)
            os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset', required=True)
    parser.add_argument("-m", '--weight_model', default='CONST')
    parser.add_argument('--model_prefix', default='1_trainset_')
    parser.add_argument('--size', type=int, default=1e6, help="Number of RR sets to generate")

    args = parser.parse_args()
    budgets = [20, 50, 100]
    evaluate(args.dataset, args.weight_model, args.model_prefix, budgets, args.size)

    # epoch vs coverage
    # budgets = [10, 20]
    # evaluate_with_epoch(args.dataset, args.network_type, args.model_prefix, budgets)
