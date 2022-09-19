# [RE] OPIM
Code source: https://github.com/tangj90/OPIM  
Paper link: https://dl.acm.org/doi/pdf/10.1145/3183713.3183749

## Build
    make clean & make -j4

## Preprocess
Generate data structure for OPIM.
```sh
example:
python preprocess.py -d BrightKite -m CONST
```

## Test
```sh
# Parameters
# -dataset: path to the dataset directory
# --weight_model: edge weight model defined for influence propagation
# Other params please refer to test.py

# Quick Start with default setting:
python test.py -d $dataset --weight_model $weight_model
```
**NOTE**: Please refer to source code for more details

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
