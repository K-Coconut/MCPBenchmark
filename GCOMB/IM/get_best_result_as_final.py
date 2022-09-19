import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="youtube")
parser.add_argument("--weight_model", default="WC")
parser.add_argument("-K", type=int, default=20)
args = parser.parse_args()

dataset = args.dataset
weight_model = args.weight_model
K = args.K

base_dir = os.path.join("GraphSAGE-master", "real_data", dataset, weight_model, "test")

best_val = 0.
for f in os.listdir(base_dir):
    match = re.match(f"_epoch_(\d+)_reward_RL_budget_{K}*", f)
    if not match:
        continue
    reward = float(open(os.path.join(base_dir, f), 'r').read())
    if reward > best_val:
        print(f, reward)
        best_val = reward
print(f"best reward: {best_val}")

with open(os.path.join(base_dir, f"best_reward_RL_budget_{K}.txt"), 'w') as f:
    f.write(f"{best_val}")