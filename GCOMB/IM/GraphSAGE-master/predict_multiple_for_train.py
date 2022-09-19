import os
import pickle
import sys

dataset = sys.argv[1]
weight_model = sys.argv[2]

budget = 50
sampling_freq = 0.003
input_dir = os.path.join("real_data", dataset, weight_model, "train") + "/"
print(input_dir)
interp_file = os.path.join('interpolator', weight_model,
                           f'interpolate_budget_percentage_real_budget_{dataset}{budget}.pkl')
dict_interpolate = pickle.load(open(interp_file, 'rb'))
num_node = int(open(os.path.join(input_dir, "attribute.txt"), 'r').read().split('\n')[0].split('=')[1])
bud_mul_fac = int(dict_interpolate(num_node))
command = f"python -m graphsage.supervisedPredict --train_prefix {input_dir} --num_k {budget} --model graphsage_meanpool --neighborhood_sampling {sampling_freq} --bud_mul_fac {bud_mul_fac}  --sigmoid --weight_model {weight_model}"
print("command ", command)
os.system(command)
