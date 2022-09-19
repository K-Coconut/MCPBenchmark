import os
import argparse

INFLUENCE_EVALUATOR = '../IMEvaluation/evaluate'


def evaluate(dataset, budgets, weight_model, epsilon, size=1e6, num=1):
    graph_file = os.path.join("..", "data", dataset, "IM", weight_model, "edges.txt")
    os.makedirs(os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model), exist_ok=True)
    format_eps = "%.1f" % epsilon
    for n in range(num):
        seed_file = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model, "multi_iter",
                                 f'budget%d_iter_{n}_seeds_IC_{format_eps}.txt')
        output_file = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model,
                                   f'coverage_%d_n_iter_{n}.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        budgets = [str(i) for i in budgets]
        klist = ",".join(budgets) + ","
        command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file} -output {output_file} -graphFile {graph_file} -klist {klist} -size {size}'
        print(command)
        os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True)
    parser.add_argument("-m", '--weight_model', required=True)
    parser.add_argument('--size', type=int, default=1e6, help="Number of RR sets to generate")
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.5, help="default=0.5")
    args = parser.parse_args()
    budgets = [20, 50, 100, 150, 200]
    evaluate(args.dataset, budgets, args.weight_model, args.eps, args.size, args.num)
