# A Benchmark Study of Deep-RL Methods for Maximum Coverage over Graphs
Zhicheng Liang, Yu Yang, Xiangyu Ke, Xiaokui Xiao, Yunjun Gao

## **Our FUll-VERSION paper can be found at** [HERE](full_version_MCPBenchmarks.pdf)

The table below lists the methods we include in the benchmark study.


| Methods | Category | Problem Solved | Paper |
|:-:|:-:|:-:|:-:|
|[S2V-DQN][s2v]| ML-based | MCP | [Learning Combinatorial Optimization Algorithms over Graphs](https://arxiv.org/abs/1704.01665)|
|[GCOMB][gcomb]| ML-based | MCP & IM | [Learning Heuristics over Large Graphs via Deep Reinforcement Learning](https://arxiv.org/abs/1903.03332)|
|[RL4IM][rl4im]| ML-based | IM | [Contingency-Aware Influence Maximization: A Reinforcement Learning Approach](https://arxiv.org/abs/2106.07039)|
|[Geometric-QN][gqn]| ML-based | IM | [Influence maximization in unknown social networks: Learning Policies for Effective Graph Sampling](https://arxiv.org/abs/1907.11625)|
|[LeNSE][LeNSE]| ML-based | MCP & IM | [LeNSE: Learning To Navigate Subgraph Embeddings for Large-Scale Combinatorial Optimisation](https://arxiv.org/abs/2205.10106)|
| Normal Greedy | Algorithmic | MCP | - |
| Lazy Greedy | Algorithmic | MCP | [Cost-effective outbreak detection in networks](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf) |
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
***NOTE***: Except for Lazy Greedy and IMDiscount, we sourced the code for all other methods from the original authors. We didn't directly fork the projects as we needed to modify the code to address MCP and IM problems, and also to remove extraneous files. Therefore, we uploaded a new version for simplicity. Readers can access the source code through the links provided in the table above.

Given that the original methods were not explicitly designed for our use-cases, we had to adapt them. For example, S2V-DQN was initially aimed at solving problems like MVC, MC, and TSP, whereas RL4IM targeted the contingency-aware IM issue. We minimally modified these codes to suit MCP or IM problems, aiming for minimal alteration while maximizing performance and replicability. For any code differences, please refer to the specific subdirectory of each project.

# Start Up
### Prerequisites
The experiments are tested under these systems:
* macOS Monterey 12.0.1, Apple clang version 12.0.0
* Ubuntu 18.04.6 LTS, gcc version 7.5.0

### Build
We recommend creating a new Anaconda environment for each Python-based project and downloading all the requirements with conda. Additionally, some projects provide a Docker file to build a container for executing the programs. Please refer to the instructions under each subdirectory to build the projects.

## 1. Preprocessing
### Datasets
We performed extensive experiments on a wide range of datasets, including real graphs of different scales. We also trained and tested RL4IM and Geometric-QN on synthetic graphs, which were generated through scripts under their respective subdirectories.

Download all real-world datasets and then process them into edge files using the provided script.

#### Real datasets
```
bash download_datasets.sh
```
#### synthetic datasets
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

