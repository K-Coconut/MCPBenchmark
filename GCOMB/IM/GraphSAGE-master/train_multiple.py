import os
import sys

dataset = sys.argv[1]
weight_model = sys.argv[2]

data_dir = os.path.join("real_data", dataset, weight_model, "train")
command = f"python -m graphsage.supervised_train --train_prefix {data_dir} --model graphsage_meanpool --sigmoid -dataset {dataset} --epochs 1000 --weight_model {weight_model}"
print(command)
os.system(command)
