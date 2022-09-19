# [RE] Discount Algorithms
# Compile
```sh
make clean & make -j4
```

# Execute
```sh
# -d: specify dataset
# -m: edge weight model, TV/CONST/WC/LND
# -a: SingleDiscount/DDiscount, default DDiscount
# -n: number of repeated iterations, default 3.
example: 
python test.py -d BrightKite -n 3 -m WC
```


## Influence Spread
Calculate the influence spread with the tool IMEvaluation.
```shell
python influence_estimation.py -d $dataset -m $weight_model 
```

## Output
Seed files, runtime and memory usage log files would be generated under the directory: result/$dataset/$weight_model/

