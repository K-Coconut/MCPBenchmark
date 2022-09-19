import os
import argparse

INFLUENCE_EVALUATOR = '../../IMEvaluation/evaluate'


def evaluate_gnn(dataset, budgets, weight_model, size=1e6):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "GraphSAGE-master", "real_data", dataset, weight_model, "test")
    graph_file = os.path.join(DATA_DIR, 'edges.txt')

    seed_file_pattern = os.path.join(DATA_DIR, "_sup_GS_sol.txt_%d_nbs_0.003")
    output_file_pattern = os.path.join(DATA_DIR, 'gnn_coverage_%d.txt')
    budgets = [str(i) for i in budgets]
    klist = ",".join(budgets) + ","
    command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file_pattern} -output {output_file_pattern} -graphFile {graph_file} -klist {klist} -size {size}'
    print(command)
    os.system(command)


def evaluate(dataset, budgets, weight_model, size=1e6):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "GraphSAGE-master", "real_data", dataset, weight_model, "test")
    graph_file = os.path.join(DATA_DIR, 'edges.txt')

    seed_file_pattern = os.path.join(DATA_DIR, "-result_RL_%d_nbs_0.003")
    output_file_pattern = os.path.join(DATA_DIR, 'coverage_%d.txt')
    budgets = [str(i) for i in budgets]
    klist = ",".join(budgets) + ","
    command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file_pattern} -output {output_file_pattern} -graphFile {graph_file} -klist {klist} -size {size}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset')
    parser.add_argument("-m", "--weight_model", default='WC')
    parser.add_argument("-s", '--size', type=int, default=1e6, help="Number of RR sets to generate")
    args = parser.parse_args()

    budgets = [20, 50, 100, 150, 200]
    evaluate(args.dataset, budgets, weight_model=args.weight_model, size=args.size)
    # evaluate_gnn(args.dataset, budgets, weight_model=args.weight_model, size=args.size)