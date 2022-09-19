import os
import argparse
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="youtube")
parser.add_argument('--weight_model', default="WC")
args = parser.parse_args()

dataset = args.dataset
weight_model = args.weight_model
FIX_VALIDATE_BUDGET = 20

res_dir = os.path.join("GraphSAGE-master/real_data", dataset, weight_model, "validation")
output_dir = os.path.join(f"trained_model_MC_{weight_model}", dataset, "model_log")

best_val = 0.
best_epoch = 0
epoch_val = defaultdict(dict)
for f in os.listdir(res_dir):
    matcher = re.match("_epoch_(\d+)_reward*", f)
    if not matcher:
        continue
    curr_epoch = int(matcher.group(1))
    curr_val = float(open(os.path.join(res_dir, f), 'r').read())
    epoch_val[curr_epoch] = curr_val

epoch_val = sorted(epoch_val.items(), key=lambda x:x[0])

epoch_to_model = {}
for f in os.listdir(output_dir):
    matcher = re.match("model_epoch_(\d+).txt", f)
    if not matcher:
        continue
    epoch = int(matcher.group(1))
    epoch_to_model[epoch] = open(os.path.join(output_dir, f), 'r').read()

for curr_epoch, curr_val in epoch_val:
    if curr_val > best_val:
        best_epoch = curr_epoch
        best_val = curr_val

    with open(os.path.join(output_dir, f"curr_epoch_{curr_epoch}_best_model.txt"), 'w') as f:
        f.write(f"{epoch_to_model[best_epoch]}")

