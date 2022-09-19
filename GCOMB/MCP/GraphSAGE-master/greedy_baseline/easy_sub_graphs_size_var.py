import os
import sys
import time
from pathlib import Path

dataset = sys.argv[1]
budgets = [20, 50, 100, 150, 200]
for budget in budgets:

    for size_var in [50, 80, 90, 99]:
        graph_path = f"../real_data/{dataset}/train{size_var}/_bp-edges.txt"
        seeds = graph_path + ".greedy" + str(budget)
        reward = graph_path + ".greedy_reward" + str(budget)
        command = f"python greedy_old.py {graph_path} {budget} {seeds} {reward}"
        st = time.time()
        print(command)
        os.system(command)
        elapsed_time = time.time() - st
        with open(f"{Path(seeds)}_time.txt", "w") as f:
            f.write(f"{elapsed_time}")
