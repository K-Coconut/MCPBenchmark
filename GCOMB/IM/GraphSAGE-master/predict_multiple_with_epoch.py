import os
import pickle
import argparse

import sys
import re
from pathlib import Path
from collections import defaultdict

dataset = sys.argv[1]
weight_model = sys.argv[2]
mode = sys.argv[3]
trained_model_to_use = 'youtube' if weight_model != 'LND' else dataset
sampling_freq = 0.003

output_path = "real_data/{}/{}/{}/".format(dataset, weight_model, mode)

budget_to_trained_epochs = defaultdict(set)
for file in os.listdir("interpolator/%s/" % weight_model):
    matcher = re.match('interpolate_budget_percentage_real_budget_{}(\\d+).pkl'.format(dataset), file)
    if matcher and int(matcher.group(1)) < 880:
        budget_to_trained_epochs[int(matcher.group(1))] = set()

trained_epochs = set(
    [int(i[11:-6]) for i in os.listdir('%s%ssupervisedTrainedModel_MC_marginal' % (weight_model, trained_model_to_use)) if i.endswith('.index')])
for file in os.listdir('real_data/%s/%s/%s/' % (dataset, weight_model, mode)):
    matcher = re.match('_epoch_(\\d+)_embeddings.npy_(\\d+)_nbs_0.003.pickle', file)
    if not matcher: continue
    trained_epoch = int(matcher.group(1))
    budget = int(matcher.group(2))
    budget_to_trained_epochs[budget].add(trained_epoch)

budget_to_trained_epochs = {k: v for k, v in sorted(budget_to_trained_epochs.items(), key=lambda x: x[0])}
task_dict = {}
for budget in budget_to_trained_epochs:
    # test for every epoch of GNN
    # task_dict[budget] = list(sorted(trained_epochs - budget_to_trained_epochs[budget]))
    # test for 1000-epoch GNN
    task_dict[budget] = [1000]

print("predicting on budgets and epochs: ", task_dict)


for budget in task_dict:
    for epoch_to_predict in task_dict[budget]:
        if Path(output_path + '_epoch_{}_embeddings.npy_{}_nbs_0.003.pickle'.format(epoch_to_predict, budget)).exists():
            print('File _epoch_{}_embeddings.npy_{}_nbs_0.003.pickle exists, continue'.format(epoch_to_predict, budget))
            continue
        file = open('interpolator/{}/interpolate_budget_percentage_real_budget_{}{}.pkl'.format(weight_model, dataset, budget), 'rb')
        dict_interpolate = pickle.load(file)
        num_node = int(
            open('./real_data/{}/{}/{}/attribute.txt'.format(dataset, weight_model, mode), 'r').read().split('\n')[0].split(
                '=')[1])
        bud_mul_fac = int(dict_interpolate(num_node) + 1) * 1.1
        command = "sh ./supervisedPredict.sh  ./real_data/%s/%s/%s/ %d %f %f %s %d %s" %(dataset, weight_model, mode, budget, sampling_freq, bud_mul_fac, weight_model, epoch_to_predict, trained_model_to_use)
        print("command ", command)
        os.system(command)
