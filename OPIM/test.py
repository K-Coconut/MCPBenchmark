import argparse
import os
from pathlib import Path

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--dataset", required=True)
   parser.add_argument("-m", "--weight_model", required=True)
   parser.add_argument("-n", "--n_iter", type=int, default=5)
   parser.add_argument("--mode", type=int, default=2)
   parser.add_argument("--eps", type=float, default=0.1)
   args = parser.parse_args()

   budgets = [20, 50, 100, 150, 200]
   mode_name_dict = {0: 'vanilla', 1: 'lastBound', 2: 'minBound'}
   for k in budgets:
      for n in range(args.n_iter):
         format_eps = "%.6f" % args.eps
         output_file = os.path.join("result", f"{args.dataset}_{args.weight_model}_opim-c_{mode_name_dict[args.mode]}_k{k}_load_eps_{format_eps}_n_iter_{n}")
         if Path(output_file).exists():
            print(f"{output_file} exists, continue")
            continue
         command = f"./OPIM1.1.o -func=1 -gname={args.dataset} -alg=opim-c -mode={args.mode} -seedsize={k} -pdist=load -wmodel={args.weight_model} -eps={args.eps} -n_iter={n}"
         print(command)
         os.system(command)