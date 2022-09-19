import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True)
args = parser.parse_args()

dataset = args.dataset
output_dir = 'result/{}/'.format(dataset)
os.makedirs(output_dir, exist_ok=True)
sampling_neighborhood = 0.75
for budget in [20, 50, 100, 150, 200]:
    graph_path = "./GraphSAGE-master/real_data/{}/test/_bp".format(dataset)
    n_nodes = int(
        open('./GraphSAGE-master/real_data/{}/test/attribute.txt'.format(dataset), 'r').readline().strip().split(
            '=')[1])

    rl_data = open(graph_path + '-reward_RL{}_nbs{}'.format(budget, sampling_neighborhood), 'r').readlines()
    coverage = float(rl_data[0].replace('\n', '')) / n_nodes * 2
    rl_time_taken = float(rl_data[1].replace('\n', '').split(':')[1])

    rl_prep_time = float(
        open(graph_path + '_num_k_{}_time_nbs{}.txt'.format(budget, sampling_neighborhood), 'r').readlines()[
            0].split('RL_PREP_TIME_')[1].replace('\n', ''))
    total_gcomb_time = rl_prep_time + rl_time_taken
    print("budget:{} coverage:{} total time:{} rl prep time: {} rl time: {}".format(budget, coverage,
                                                                                    total_gcomb_time, rl_prep_time,
                                                                                    rl_time_taken))
    result = '{}/budget{}.txt'.format(output_dir, budget)
    out_gcomb_quality_time_sample = open(result, 'w')
    out_gcomb_quality_time_sample.write(f"{coverage}\t{total_gcomb_time}")
