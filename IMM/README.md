# [RE] IMM
Code source: https://sourceforge.net/projects/im-imm/  
Paper link: https://doi.org/10.1145/2723372.2723734

## Compile
    make clean & make -j4

## Execute
```sh
# Parameters
# -dataset: path to the dataset directory
# -epsilon: a double value for epsilon, is set as 0.5 in our experiment
# -k: number of selected nodes
# --weight_model: edge weight model defined for influence propagation

# Quick Start with default setting:
python test.py -d $dataset -k $budget --weight_model $weight_model
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
