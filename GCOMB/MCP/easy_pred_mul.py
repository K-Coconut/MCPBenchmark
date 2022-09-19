import os
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', type=str, required=True)
args = parser.parse_args()
dataset = args.dataset
budget_list = [20, 50, 100, 150, 200]

for i in range(0,1):
    model_path = open('bestRlModel.txt','r').read()
    base_dir = './GraphSAGE-master/real_data/{}/test/'.format(dataset)
    for sampling_neighborhood in [0.75]:
        for budget in budget_list:

            if Path("{}_bp-reward_RL{}_nbs{}".format(base_dir, budget, sampling_neighborhood)).exists():
                print("budget {} processed".format(budget))
                continue
            graph_path = "./GraphSAGE-master/real_data/{}/test/_bp".format(dataset)
            command = "python get_output.py " + graph_path + " " +model_path.replace("\n","") + " " + str(budget) +" " + str(sampling_neighborhood) + " None"
            print(command)
            os.system(command)
