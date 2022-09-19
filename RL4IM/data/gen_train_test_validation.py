import os
import networkx as nx
import random
import numpy as np

NUMBER_OF_VALIDATION_GRAPHS = 5
NODE_VAR = 20
M = 3
P = 0.05

# Study impact of different size of training subgraphs
# Validation graphs
for graph_size in [80, 100, 200, 300, 500, 700, 1000]:
    for i in range(NUMBER_OF_VALIDATION_GRAPHS):
        output_path = "synthetic/study_subgraph_size/validation/graphsize_%d/G%d/" % (
            graph_size, i)
        os.makedirs(output_path, exist_ok=True)
        seed = 200000 + i
        np.random.seed(seed)
        cur_n = graph_size
        max_node_num = cur_n + NODE_VAR
        cur_n += np.random.choice(range(-NODE_VAR, NODE_VAR + 1, 1))
        g = nx.powerlaw_cluster_graph(
            n=cur_n, m=M, p=P, seed=seed).to_directed()
        with open(output_path + 'edges.txt', 'w') as f:
            for edge in g.edges():
                f.write("%d %d %f\n" %
                        (edge[0], edge[1], 1 / g.in_degree(edge[1])))
        with open(output_path + 'attribute.txt', 'w') as f:
            f.write(f"n={g.number_of_nodes()}\nm={g.number_of_edges()}")

# Training graphs
FIX_NUMBER_OF_TRAIN_GRAPHS = 200
for graph_size in [80, 100, 200, 300, 500, 700, 1000]:
    for i in range(FIX_NUMBER_OF_TRAIN_GRAPHS):
        output_path = "synthetic/study_subgraph_size/trainset/graphsize_%d/G%d/" % (
            graph_size, i)
        os.makedirs(output_path, exist_ok=True)
        seed = i
        np.random.seed(seed)
        cur_n = graph_size
        max_node_num = cur_n + NODE_VAR
        cur_n += np.random.choice(range(-NODE_VAR, NODE_VAR + 1, 1))
        g = nx.powerlaw_cluster_graph(
            n=cur_n, m=M, p=P, seed=seed).to_directed()
        with open(output_path + 'edges.txt', 'w') as f:
            for edge in g.edges():
                f.write("%d %d %f\n" %
                        (edge[0], edge[1], 1 / g.in_degree(edge[1])))
        with open(output_path + 'attribute.txt', 'w') as f:
            f.write(f"n={g.number_of_nodes()}\nm={g.number_of_edges()}")

# # Study impact of different size of training samples
FIX_GRAPH_SIZE = 200
for i in range(NUMBER_OF_VALIDATION_GRAPHS):
    output_path = "synthetic/study_sample_size/validation/G%d/" % i
    os.makedirs(output_path, exist_ok=True)
    seed = 200000 + i
    np.random.seed(seed)
    cur_n = FIX_GRAPH_SIZE
    max_node_num = cur_n + NODE_VAR
    cur_n += np.random.choice(range(-NODE_VAR, NODE_VAR + 1, 1))
    g = nx.powerlaw_cluster_graph(n=cur_n, m=M, p=P, seed=seed).to_directed()
    with open(output_path + 'edges.txt', 'w') as f:
        for edge in g.edges():
            f.write("%d %d %f\n" %
                    (edge[0], edge[1], 1 / g.in_degree(edge[1])))
    with open(output_path + 'attribute.txt', 'w') as f:
        f.write(f"n={g.number_of_nodes()}\nm={g.number_of_edges()}")


# we study sample size within the range of 10 to 1000
NUMBER_OF_TRAIN_GRAPHS = 1000
for i in range(NUMBER_OF_TRAIN_GRAPHS):
    output_path = "synthetic/study_sample_size/trainset/G%d/" % i
    os.makedirs(output_path, exist_ok=True)
    seed = i
    np.random.seed(seed)
    cur_n = FIX_GRAPH_SIZE
    max_node_num = cur_n + NODE_VAR
    cur_n += np.random.choice(range(-NODE_VAR, NODE_VAR + 1, 1))
    g = nx.powerlaw_cluster_graph(n=cur_n, m=M, p=P, seed=seed).to_directed()
    with open(output_path + 'edges.txt', 'w') as f:
        for edge in g.edges():
            f.write("%d %d %f\n" %
                    (edge[0], edge[1], 1 / g.in_degree(edge[1])))
    with open(output_path + 'attribute.txt', 'w') as f:
        f.write(f"n={g.number_of_nodes()}\nm={g.number_of_edges()}")
