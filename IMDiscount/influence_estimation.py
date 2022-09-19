import os
import argparse

INFLUENCE_EVALUATOR = '../IMEvaluation/evaluate'


def evaluate(dataset, budgets, algorithm, weight_model, size=1e6):
    graph_file = os.path.join("..", "data", dataset, "IM", weight_model, "edges.txt")
    os.makedirs(os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model), exist_ok=True)
    seed_file = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model,
                             f'{algorithm}_budget%d_seeds.txt')
    output_file = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model,
                               f'{algorithm}_coverage_%d.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    budgets = [str(i) for i in budgets]
    klist = ",".join(budgets) + ","
    command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file} -output {output_file} -klist {klist} -graphFile {graph_file} -size {size}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True)
    parser.add_argument('-m', '--weight_model', required=True)
    parser.add_argument('--size', type=int, default=1e6, help="Number of RR sets to generate")
    parser.add_argument("--algorithm", default="DDiscount", help="DDiscount/SingleDiscount, default=DDiscount")
    args = parser.parse_args()
    budgets = [20, 50, 100, 150, 200]
    evaluate(args.dataset, budgets, args.algorithm, args.weight_model, args.size)
