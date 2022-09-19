# Influence Estimation

A tool to estimate the influence spread of solution sets.

## Acknowledge
Adapted from [IMM](https://sourceforge.net/projects/im-imm/).

## Compile
```sh
make clean && make
```

## Execute
We integrated the tool in Python scripts under each sub-project, execute these scripts to calculate the influence spread.  
```sh
python influence_estimation.py -d $dataset ..
```

You can also execute the binary file directly.
```shell
 # -size: Required. Number of RR sets to generate, the larger the more accurate. 1e5 is adequate to most cases.
 # -seedFile: Required, the path of the file containing solution set 
 # -outputFile: Required, the output path of the result file.
 # -graphFile: Required, the path of the edges file
 # -klist: Option, a series of budgets to calculate. The paths of seedFile and outputFile need to be a pattern rather than real file paths.
 # -seed_random: Option, random seed
 
example:
./evaluate -seedFile ./result/seed10.txt -outputFile ./result/coverageq0.txt  -graphFile ../data/BrightKite/IM/WC/edges.txt -size 1000000

# use the klist param to calculate coverage of all the seed files under the directory.
./evaluate -seedFile ./result/seed%d.txt -outputFile ./result/coverage%d.txt  -graphFile ../data/BrightKite/IM/WC/edges.txt -klist 20,50,100,150,200, -size 1000000  
```

