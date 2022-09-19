import os
import re
from pathlib import Path
from collections import defaultdict
import argparse

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset')
args.add_argument('--weight_model', default="WC")
args.add_argument('--mode', required=True, help="train or validation or test")
parser = args.parse_args()

dataset = parser.dataset
trained_model = parser.trained_model
sampling_freq = 0.003
weight_model = parser.weight_model
mode = parser.mode

assert mode in ['test', 'train', 'validation']

output_path = "GraphSAGE-master/real_data/{}/{}/{}/".format(dataset, weight_model, mode)

budget_to_trained_epochs = defaultdict(set)
task_dict = {}

for file in os.listdir("GraphSAGE-master/interpolator/%s/" % weight_model):
    matcher = re.match('interpolate_budget_percentage_real_budget_{}(\\d+).pkl'.format(dataset), file)

trained_epochs = set([int(i[12:-4]) for i in os.listdir('trained_model_MC_%s/%s/model_log/' % (weight_model, dataset)) if i.startswith('model')])
for i in trained_epochs.copy():
    if i > 10000 and i % 10000 != 0 or i % 1000 != 0:
        trained_epochs.remove(i)

for file in os.listdir('GraphSAGE-master/real_data/youtube/%s/%s/' % (weight_model, mode)):
    matcher = re.match('_epoch_(\\d+)-result_RL_(\\d+)_nbs_0.003', file)
    if not matcher: continue
    trained_epoch = int(matcher.group(1))
    budget = int(matcher.group(2))
    budget_to_trained_epochs[budget].add(trained_epoch)

budget_to_trained_epochs = {k: v for k, v in sorted(budget_to_trained_epochs.items(), key=lambda x: x[0])}

for budget in budget_to_trained_epochs:
    task_dict[budget] = list(sorted(trained_epochs - budget_to_trained_epochs[budget]))

print("predicting on budgets and epochs: ", task_dict)


for budget in task_dict:
    for epoch_to_load in task_dict[budget]:
        # using 1000-epoch GNN embedding file
        if Path(output_path + '_epoch_{}-result_RL_{}_nbs_0.003'.format(epoch_to_load, budget)).exists():
            print('File _epoch_{}-result_RL_{}_nbs_0.003 exists, continue'.format(epoch_to_load, budget))
            continue
        # predict on training dataset
        command = "python get_output.py -p ./GraphSAGE-master/real_data/%s/%s/%s/ --weight_model %s -k %d -f %f -m Notnone -e %d" % (dataset, weight_model, mode, weight_model, budget, sampling_freq, epoch_to_load)
        print(command)
        os.system(command)