# [RE] LeNSE

## Acknowledge
Code Source: https://github.com/davidireland3/LeNSE
Paper link: https://arxiv.org/abs/2205.10106

## Build
```sh
bash build.sh
```

## Adaption
1. For simplicity, we remove unused files and annotated codes; reorganize the project and output path
2. More parameters added to make experiments more flexible
3. Simplify the codes by removing some unnecessary/unused variables to reduce memory consumption and speed up.
4. Improve preprocessing efficiency (from >48 hours to minutes):
   1. Evaluate influence with RR sets.
   2. Replace Python-version IMM with a C++-version one.
   3. Optimize data generation process while maintaining underlying logic.


## Preprocess
We follow the instructions from source code to preprocess the input data.
More details please refer to the [source code](https://github.com/davidireland3/LeNSE). 
```sh
# preprocess for MCP
cd MVC
bash data_link.sh $dataset $weight_model
bash pre.sh $dataset

# preprocess for IM
cd IM/preprocess
bash pre.sh $dataset $weight_model
```

1. Execute script to generate test dataset.
```sh
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
Our trained model can be downloaded from [HERE](https://drive.google.com/drive/folders/1pOc6dj0oprQYq4wvu6BKMYvv4sqEXA3i?usp=drive_link).
Put the model under ./IM/data/youtube/ or ./MVC/data/BrightKite/


## Test
```sh
# MCP
cd MCP
bash test.sh $dataset 

# IM
cd IM
bash test.sh $dataset $weight_model
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
* Format: $coverage $runtime $memory
