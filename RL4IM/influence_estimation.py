import os
import argparse

INFLUENCE_EVALUATOR = '../IMEvaluation/evaluate'


def evaluate_synthetic(budgets, size=1e6, num_graphs=10):
    DATA_DIR = "data/"
    for i in range(num_graphs):
        graph_file = os.path.join(os.path.dirname(__file__), DATA_DIR, 'synthetic', 'G%d' % i, 'edges.txt')
        seed_file = os.path.join(os.path.dirname(__file__), DATA_DIR, 'synthetic', 'G%d' % i, 'seed%d.txt')
        output_file = os.path.join(os.path.dirname(__file__), DATA_DIR, 'synthetic', 'G%d' % i, 'coverage_%d.txt')
        budgets = [str(i) for i in budgets]
        klist = ",".join(budgets) + ","
        command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file} -output {output_file} -graphFile {graph_file} -klist {klist} -size {size}'
        print(command)
        os.system(command)


def evaluate(dataset, budgets, weight_model, size=1e6):
    DATA_DIR = "../data/"
    graph_file = os.path.join(os.path.dirname(__file__), DATA_DIR, dataset, 'IM', weight_model, 'edges.txt')

    seed_file_pattern = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model, 'seed%d.txt')
    output_file_pattern = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model, 'coverage_%d.txt')
    os.makedirs(os.path.dirname(output_file_pattern), exist_ok=True)
    budgets = [str(i) for i in budgets]
    klist = ",".join(budgets) + ","
    command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file_pattern} -output {output_file_pattern} -graphFile {graph_file} -klist {klist} -size {size}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset')
    parser.add_argument("-m", "--weight_model", required=True)
    parser.add_argument("-s", '--size', type=int, default=1e6, help="Number of RR sets to generate")
    args = parser.parse_args()

    budgets = [20, 50, 100, 150, 200]
    evaluate(args.dataset, budgets, weight_model=args.weight_model, size=args.size)

    # evaluate on synthetic graphs
    # budgets = list(range(1, 20)) + [30, 40, 50, 100, 150]
    # evaluate_synthetic(budgets, args.size)
