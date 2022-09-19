import os
from pathlib import Path
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True)
args = parser.parse_args()
dataset = args.dataset
mode = "test"
sampling_rate = 0.75
budget_list = [20, 50, 100, 150, 200]

base_dir = "./real_data/{}/{}/".format(dataset, mode)
for budget in budget_list:

    if Path("{}_bp_num_k_{}_time_nbs{}.txt".format(base_dir, budget, sampling_rate)).exists():
        print("budget {} processed, continue".format(budget))
        continue

    num_node = int(
        open('{}attribute.txt'.format(base_dir), 'r').read().split('\n')[0].split(
            '=')[1])
    interp_file = 'interpolator/interpolate_budget_percentage_real_budget_{}{}.pkl'.format(dataset, budget)
    file = open(interp_file, 'rb')
    dict_interpolate = pickle.load(file)
    print("nodes in graph", num_node)
    use_upto = dict_interpolate(num_node)
    print("budget , use up to ", budget, use_upto)
    command = "sh ./supervisedPredict.sh ./real_data/{}/".format(dataset) + str(mode) + "/_bp" + " " + str(
        budget) + " " + str(use_upto) + " " + str(sampling_rate)
    print(command)
    os.system(command)
