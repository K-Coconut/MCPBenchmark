import argparse
import os


def parse(dataset, weight_model, mode, budget, eps, num=5):
   format_eps = "%.6f" % eps
   reward = 0.
   elasped_time = 0.
   memory_usage = 0.
   for n in range(num):
      stat_file_path = os.path.join("result", f"{dataset}_opim-c_{mode}_{weight_model}_k{budget}_load_eps_{format_eps}_n_iter_{n}")
      elasped_time += float(open(stat_file_path, "r").read().split('\n')[1].split(": ")[1])
      memory_usage += float(open(stat_file_path, "r").read().split('\n')[6].split(": ")[1])
      reward += float(open(os.path.join("result", dataset, weight_model, f"coverage_{budget}_n_iter_{n}.txt")).read())
   output_path = os.path.join(OUTPUT_DIR, f"budget{budget}.txt")
   reward /= num
   elasped_time /= num
   memory_usage /= num
   with open(output_path, "w") as f:
      print(f"budget {budget}: reward: {reward} time: {elasped_time} memory: {memory_usage} MB")
      f.write(f"{reward} {elasped_time} {memory_usage}")


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--dataset", required=True)
   parser.add_argument("-m", "--weight_model", required=True)
   parser.add_argument('--num', type=int, default=5)
   parser.add_argument("--mode", type=int, default=2, help="integer, default=2, 0: 'vanilla', 1: 'lastBound', 2: 'minBound'")
   parser.add_argument("--eps", type=float, default=0.1, help="default=0.1")
   args = parser.parse_args()

   OUTPUT_DIR = os.path.join("result", args.dataset, args.weight_model)
   os.makedirs(OUTPUT_DIR, exist_ok=True)
   
   mode_name_dict = {0: 'vanilla', 1: 'lastBound', 2: 'minBound'}
   budgets = [20, 50, 100, 150, 200]
   for k in budgets:
      parse(args.dataset, args.weight_model, mode_name_dict[args.mode], k, args.eps, args.num)
