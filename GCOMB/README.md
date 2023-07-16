# [RE] GCOMB

## Acknowledge
Code Source: https://github.com/idea-iitd/GCOMB  
Paper link: https://arxiv.org/abs/1903.03332

## Build
```sh
bash build.sh
```

## Adaption
1. For simplicity, we remove unused files and annotated codes; reorganize the project and output path
2. More parameters added to make experiments more flexible
3. Simplify the codes by removing some unnecessary/unused variables to reduce memory consumption and speed up.
4. Change format of the graph from json to text to reduce massive memory consumption caused by IO, etc. 

## Preprocess
We follow the instructions from source code to preprocess the input data.
More details please refer to the [source code](https://github.com/idea-iitd/GCOMB). 
```sh
# preprocess for MCP
cd MCP
bash pre_process.sh $dataset

# preprocess for IM
cd IM
bash pre_process.sh $dataset $weight_model
```

You can also skip the preprocess procedure by using the our noise interpolator directly.
1. Download interpolators from [HERE](https://drive.google.com/drive/folders/1V0TcKcC5AW_CZUhKwN7wTHgAVbtX4pg1?usp=drive_link) and then put them under GraphSAGE-master directory.
2. Execute script to generate test dataset.
    ```shell
    # MCP
    cd MCP/GraphSAGE-master/real_data
    python create_bp.py $dataset
   
    # IM
    cd IM/GraphSAGE-master/real_data
    python convto_nx.py -d $DATASET -m $WEIGHT_MODEL -t test
    ```
## Train
```sh
# MCP
cd MCP
bash train.sh

# IM
cd IM
bash train.sh $weight_model
```
Our trained model can be downloaded from [HERE](https://drive.google.com/drive/folders/1V0TcKcC5AW_CZUhKwN7wTHgAVbtX4pg1?usp=drive_link).
Put the RL model (trained_model_MC*) under ./IM or ./MCP, and the GCN model (*supervisedTrainedModel_MC_marginal) under GraphSAGE-master.


## Test
```sh
# MCP
cd MCP
bash test.sh $dataset 

# IM
cd IM
bash test.sh $dataset $weight_model
```

## Influence Spread
In IM, we estimate the influence spread with a common tool IMEvaluation.
```sh
# -d: Required, dataset
# --weight_model: Required, specify the edge weight model 
# --size: Option, number of RR sets to generate, default set as 1e6.

python influence_estimation.py -d $dataset -m $weight_model
```

## Output Format
Extract the results
```sh
# MCP
python parse_result.py -d $dataset

# IM
python parse_result.py -d $dataset --weight_model $weight_model
```
* output path: 
    result/$dataset/budget$k.txt
* Format: $coverage $runtime
