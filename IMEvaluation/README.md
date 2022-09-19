# Influence Estimation

A tool to estimate the influence spread of solution sets.

## Acknowledge

The code is adapted from [IMM](https://sourceforge.net/projects/im-imm/).  
More about Reverse Influence Sampling
techniques:  [Maximizing Social Influence in Nearly Optimal Time](https://doi.org/10.1137/1.9781611973402.70)

## Theoretical Support

First, we run Monte Carlo simulation to generate a set of $RR$ sets $\mathcal{R}$, $\mathcal{R} = \{R_1,R_2,...,R_N\}$.
For a set of vertices $S$, the degree of $S$ in $\mathcal{R}$ is denoted by $D(S)$, which is the number of $RR$ sets
containing at least one vertex in $S$. According to the linearity of expectation, $\frac{nD(S)}{N}$ is an unbiased
estimator of $I(S)$ [[1](https://doi.org/10.1137/1.9781611973402.70)]. Thus, in the polling method, $\frac{nD(S)}{N}$ is used to
approximate $I(S)$. 

## Compile

```sh
make clean && make
```

## Execute

We integrated the tool in Python scripts under each sub-project, execute these scripts to calculate the influence
spread.

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

