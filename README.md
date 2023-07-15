# Benchmark Study for MCP
Implementation for the following paper: "A Worrying Analysis of Deep-RL Methods for Maximum Coverage over Graphs: A Benchmark Study"

The table below lists the methods we include in the benchmark study.


| Methods | Category | Problem Solved | Paper |
|:-:|:-:|:-:|:-:|
|[S2V-DQN][s2v]| ML-based | MCP | [Learning Combinatorial Optimization Algorithms over Graphs](https://arxiv.org/abs/1704.01665)|
|[GCOMB][gcomb]| ML-based | MCP & IM | [Learning Heuristics over Large Graphs via Deep Reinforcement Learning](https://arxiv.org/abs/1903.03332)|
|[RL4IM][rl4im]| ML-based | IM | [Contingency-Aware Influence Maximization: A Reinforcement Learning Approach](https://arxiv.org/abs/2106.07039)|
|[Geometric-QN][gqn]| ML-based | IM | [Influence maximization in unknown social networks: Learning Policies for Effective Graph Sampling](https://arxiv.org/abs/1907.11625)|
|[LeNSE][LeNSE]| ML-based | IM & MCP | [LeNSE: Learning To Navigate Subgraph Embeddings for Large-Scale Combinatorial Optimisation](https://arxiv.org/abs/2205.10106)|
| Normal Greedy | Algorithmic | MCP | - |
| Lazy Greedy | Algorithmic | MCP | - |
| [IMM][imm] | Algorithmic | IM | [Influence Maximization in Near-Linear Time: A Martingale Approach](https://doi.org/10.1145/2723372.2723734) |
| [OPIM][opim] | Algorithmic | IM |[Online Processing Algorithms for Influence Maximization](https://dl.acm.org/doi/pdf/10.1145/3183713.3183749)|
| DegreeDiscount & SingleDiscount | Algorithmic | IM | [Efficient influence maximization in social networks](https://dl.acm.org/doi/10.1145/1557019.1557047)|

[s2v]: https://github.com/Hanjun-Dai/graph_comb_opt
[gcomb]: https://github.com/idea-iitd/GCOMB
[rl4im]: https://github.com/Haipeng-Chen/RL4IM-Contingency
[gqn]: https://github.com/kage08/graph_sample_rl
[LeNSE]: https://github.com/davidireland3/LeNSE
[imm]: https://sourceforge.net/projects/im-imm/
[opim]: https://github.com/tangj90/OPIM


## *Prologue*
***NOTE***: With the exception of Lazy Greedy and IMDiscount, we derived the codes of all the methods from the original authors. However, we did not directly fork the projects because we needed to adapt the codes to solve MCP and IM problems, and remove unnecessary files. Therefore, we uploaded a new version for simplicity. Readers can access the source code through the links provided in the table above.

Since not all methods were specifically tailored for our purposes, we had to make adaptations to solve our specific problems. For instance, S2V-DQN was originally designed for solving Minimum Vertex Cover (MVC), Maximum Cut (MC), and Traveling Salesman Problem (TSP), while RL4IM was designed for addressing the contingency-aware IM problem. Nevertheless, we were able to easily adapt these methods to solve MCP or IM problems. Our aim was to make minimal changes to the code while maximizing the performance of these methods and reproducing the results. Please refer to the subdirectory of each project and examine the source code to identify any differences.

# Start Up
### Prerequisites
The experiments are tested under these systems:
* macOS Monterey 12.0.1, Apple clang version 12.0.0
* Ubuntu 18.04.6 LTS, gcc version 7.5.0

### Build
We recommend that creating a new Anaconda environment for each python-based project, and download all the requirements with conda. Also, some projects provide a Docker file to build up a container to execute the programs.
Please refer to the instructions under each subdirectory to build the projects.
## 1. Preprocessing
### Datasets
We perform an extensive experiment over a wide range of datasets including real graphs of different scale. We also try to train and test RL4IM, Geometric-QN on synthetic graphs, these synthetic datasets are generated through scripts and stored in their subdirectories.

Download all real-world datasets and then process them into the edges files with the script. 

#### Real datasets
```
bash download_datasets.sh
```
#### synthetic datasets.
Follow instructions in README under RL4IM and Geometric-QN to generate synthetic datasets.

### Preprocess for specific Task
1. Define edge weight model, and generate specific edges files for IM problem.
```sh
# example
cd data
python preprocess.py -d BrightKite -m WC
```
2. Methods like GCOMB, LeNSE need a long pipeline of preprocessing before training and test, please execute preprocessing script under the subdirectory.

## 2. Training & Test
* Follow instructions in each project to train models and test.
* Calculate the influence spread with the tool IMEvaluation in IM problem.

