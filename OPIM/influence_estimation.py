import os
import argparse

INFLUENCE_EVALUATOR = '../IMEvaluation/evaluate'

def evaluate(dataset, budgets, weight_model, mode, eps, size=1e6, num=1):
   graph_file = os.path.join("..", "data", dataset, "IM", weight_model, "edges.txt")
   os.makedirs(os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model), exist_ok=True)
   format_eps = "%.6f" % eps
   for n in range(num):
      seed_file = os.path.join(os.path.dirname(__file__), 'result', 'seed', f"seed_{dataset}_opim-c_{mode}_{weight_model}_k%d_load_eps_{format_eps}_n_iter_{n}")
      output_file = os.path.join(os.path.dirname(__file__), 'result', dataset, weight_model, f'coverage_%d_n_iter_{n}.txt')
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
   
      budgets = [str(i) for i in budgets]
      klist = ",".join(budgets) + ","
      command = f'{INFLUENCE_EVALUATOR} -seedFile {seed_file} -output {output_file} -graphFile {graph_file} -klist {klist} -size {size}'
      print(command)
      os.system(command)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-d', "--dataset", required=True)
   parser.add_argument('--weight_model', required=True)
   parser.add_argument('--size', type=int, default=1e6, help="Number of RR sets to generate")
   parser.add_argument('--num', type=int, default=5, help="repeated times of experiments")
   parser.add_argument("--mode", type=int, default=2, help="integer, default=2, 0: 'vanilla', 1: 'lastBound', 2: 'minBound'")
   parser.add_argument("--eps", type=float, default=0.1, help="default=0.1")
   args = parser.parse_args()
   budgets = [20, 50, 100, 150, 200]
   mode_name_dict = {0: 'vanilla', 1: 'lastBound', 2: 'minBound'}
   evaluate(args.dataset, budgets, args.weight_model, mode_name_dict[args.mode], args.eps, args.size, args.num)
