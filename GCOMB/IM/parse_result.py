import os
import argparse

args = argparse.ArgumentParser()
args.add_argument('-d', '--dataset', required=True)
args.add_argument('-m', "--weight_model", required=True)
parser = args.parse_args()

dataset = parser.dataset
weight_model = parser.weight_model

base_path = os.path.join("GraphSAGE-master", "real_data", dataset, weight_model, "test")
OUTPUT_DIR = os.path.join("result", dataset, weight_model)
os.makedirs(OUTPUT_DIR, exist_ok=True)

budgets = [20, 50, 100, 150, 200]
sampling_freq = 0.003

for budget in budgets:
    rl_time_taken = float(open(os.path.join(base_path, f'-time_RL_budget{budget}_nbs_{sampling_freq}'), 'r').read())
    rl_prep_file_path = os.path.join(base_path, f'_num_k_{budget}_time.txt_{budget}_nbs_{sampling_freq}')
    rl_prep_time = float(open(rl_prep_file_path, 'r').readline().strip().split('RL_PREP_TIME_')[1])
    total_gcomb_time = rl_prep_time + rl_time_taken
    reward = float(open(os.path.join(base_path, f"coverage_{budget}.txt" ), 'r').read())
    print(f"K: {budget}\tcoverage: {reward}\ttotal time: {total_gcomb_time}\trl prep time: {rl_prep_time}\trl time: {rl_time_taken}")
    out_gcomb_quality_time_sample = open(os.path.join(OUTPUT_DIR, f"budget{budget}.txt"), 'w')
    out_gcomb_quality_time_sample.write(f"{reward} {total_gcomb_time}")
