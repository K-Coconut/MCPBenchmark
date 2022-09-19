import os
import pickle
import argparse

args = argparse.ArgumentParser()
args.add_argument("-d", "--dataset", required=True)
args.add_argument("-m", "--weight_model", required=True)
args.add_argument("--mode", default="test", help="train, validation, test")
args = args.parse_args()

dataset = args.dataset
weight_model = args.weight_model
mode = args.mode
input_prefix = os.path.join("real_data", dataset, weight_model, mode) + '/'
sampling_freq = 0.003

budgets = [20, 50, 100, 150, 200]
for budget in budgets:
    interp = os.path.join("interpolator", weight_model,
                          f"interpolate_budget_percentage_real_budget_{dataset}{budget}.pkl")
    dict_interpolate = pickle.load(open(interp, 'rb'))
    num_node = int(open(os.path.join(input_prefix, 'attribute.txt'), 'r').read().split('\n')[0].split('=')[1])
    bud_mul_fac = int(dict_interpolate(num_node) + 1) * 1.1
    command = f"python -m graphsage.supervisedPredict --train_prefix {input_prefix} --num_k {budget} " \
              f"--model graphsage_meanpool --sigmoid --neighborhood_sampling {sampling_freq} --bud_mul_fac {bud_mul_fac}" \
              f" --weight_model {weight_model}"
    print("command ", command)
    os.system(command)
