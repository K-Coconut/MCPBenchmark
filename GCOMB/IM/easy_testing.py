import os
import argparse
from pathlib import Path

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", required=True)
args.add_argument("-m", "--weight_model", required=True)
args.add_argument("--mode", default="test", help="train, validation, test")
args = args.parse_args()

dataset = args.dataset
weight_model = args.weight_model
mode = args.mode

trained_dataset = 'youtube'
output_path = "GraphSAGE-master/real_data/{}/{}/{}/".format(dataset, weight_model, mode)

budgets = [20, 50, 100, 150, 200]
sampling_freq = 0.003

print("testing on budgets: ", budgets)

for budget in budgets:

    if Path(output_path + '-result_RL_{}_nbs_0.003'.format(budget)).exists():
        print('File -result_RL_{}_nbs_0.003 exists, continue'.format(budget))
        # continue
    command = f"python get_output.py -p {output_path} -k {budget} -f {sampling_freq} -d {trained_dataset} -m best --weight_model {weight_model}"
    print(command)

    os.system(command)
