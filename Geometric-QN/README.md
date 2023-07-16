# [RE] Geometric-QN
## Acknowledge
Code source: https://github.com/kage08/graph_sample_rl  
Paper link: https://arxiv.org/abs/1907.11625

# Build
```shell
bash build.sh
```

# Train
**NOTE**: We can train the model with different combinations of datasets, more details can be found at the [paper](https://arxiv.org/abs/1907.11625). Please annotate or unannotate the code in train.py to train the model in different ways, and specify the model_prefix.
Download the training datasets from [HERE](https://drive.google.com/drive/folders/1kiWdgw04neMHuxejZeN9ihsxNaxTdkKu?usp=drive_link) and then put them under the directory *train_data*. 

```sh
# params: please refer to the code.

# Quick start with default setting:
python train.py --model_prefix 2_trainset_
```

Our trained models can be downloaded from [HERE](https://drive.google.com/drive/folders/1JIgcb1vIz-vQg9ixilgUYpkUShect0qH?usp=drive_link). Put the files under the directory *models*.

# Test
```sh
# -d: specify dataset
# --weight_model: edge weight model, TV/CONST/WC/LND
# -k: budget, default 20
# -n: number of repeated iterations, default 3
# --model_prefix: the prefix of the trained model's filename
# other params please refer to the codes

Quick start example:
python test.py -d BrightKite -k 20 --weight_model CONST --model_prefix 1_trainset_
```

## Influence Spread
Calculate the influence spread with the tool IMEvaluation.
```shell
python influence_estimation.py -d $dataset -m $weight_model 
```

## Output Format
run the script to calculate the average of the result in numerous experiments with the default settings. 
```shell
python parse_result.py -d $dataset -m $weight_model
```
* output path: result/$dataset/$weight_model/budget$k.txt
* result format: $coverage $runtime $memory
