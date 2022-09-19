# [RE] RL4IM

## Acknowledge
Code source: https://github.com/Haipeng-Chen/RL4IM-Contingency  
Paper link: https://arxiv.org/abs/2106.07039


## Installation
```sh
bash build.sh
```

## Adaption
* We modify the code to solve IM problem rather than contingency IM problem. 
* We optimize the matrix operations to improve the model's scaliability by thoudsands times, etc.

## Training
Please refer to source code to find more details about running project on Docker, HPC, etc.

1. Default command line:
```bash
python main.py --config=rl4im --env-config=basic_env --results-dir=results with lr=1e-3
```
All the default environment and method-related parameters are specified in `src/tasks/config`. You can set customized values of hyperparameters after `with` as demonstrated in the above command.

Our trained models can be downloaded from [HERE](https://drive.google.com/drive/u/0/folders/1fKXvSbwZOka_Y1AYDORvw6scZ-fKWSSO)


## Test
### Test on Real Graph
Quick Start:
1. Test
```sh
example:
python test_real_graph.py -d BrightKite --weight_model CONST
```
2. Influence Estimation
```sh
python influence_estimation.py -d BrightKite --weight_model CONST
```


### Test on Synthetic Graph
1. Generate Synthetic Graphs
```sh
cd data
python gen_synthetic_graphs.py
```
2. Test
```sh
example:
python test_synthetic.py 
```

3. Influence Estimation
```sh
python influence_estimation.py
```

## More Insights
### Performance during Training Procedure
#### 1. Get the best model so far at each epoch
```sh
# --model_prefix: sacred, sacred, sample_size_sacred, graph_size_sacred
# -p: checkpoint path
example:
python get_best_model.py --model_prefix sacred  -p 1 
```
#### 2. Run test on with the best model saved at each epoch
```sh
# --model_prefix: sacred, sacred, sample_size_sacred, graph_size_sacred
# -p: checkpoint path
example:
python test_synthetic_with_epoch.py -p 1
```

### Impact of Training Dataset Size
#### 1. Generate Synthetic Graphs
```sh
cd data
# generate synthetic graphs to study about the impact of training dataset size
python gen_train_test_validation.py
```
#### 2. Train
Modify the Input/Output path in main.py, and then train the models.
#### 2. Test
Modify the Input/Output path in test_synthetic.py, and then test on the new generated synthetic graphs.

An end-to-end pipeline version would be provided in the future.