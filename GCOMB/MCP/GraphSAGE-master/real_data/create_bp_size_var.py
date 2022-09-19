import networkx as nx
import json
from networkx.readwrite import json_graph
from networkx.algorithms import bipartite
import sys


def gen_setcover_inst(G):
    constant_add = len(G)
    print(" const add ", constant_add)
    node_list_A = list(set(G.nodes()))
    node_list_B = [x + constant_add for x in list(set(G.nodes()))]

    g = nx.Graph()
    g.add_nodes_from(node_list_A, bipartite=0)
    g.add_nodes_from(node_list_B, bipartite=1)

    for (v1, v2) in G.edges():
        if v1 == v2:
            continue
        g.add_edge(v1, v2 + constant_add)
        g.add_edge(v1 + constant_add, v2)

    set_nodes, element_nodes = bipartite.sets(g)
    set_nodes = list(set_nodes)
    element_nodes = list(element_nodes)
    print(len(set_nodes), len(element_nodes))
    return g


dataset = sys.argv[1]
train_or_test = "train"

for size_var in [50, 80, 90, 99]:
    print("size_var ", size_var)
    input_file_path = f"{dataset}/{train_or_test}{size_var}/edges.txt"
    graph_name = f"{dataset}/{train_or_test}{size_var}/_bp"
    G = nx.read_edgelist(input_file_path, nodetype=int)
    G = gen_setcover_inst(G)

    total_nodes = len(G)
    print(" total nodes ", total_nodes)
    validation_set = (0.99 * total_nodes)
    test_set = (0.99 * total_nodes)
    random_list = [i for i in range(total_nodes)]

    with open(graph_name + "-edges.txt", 'w') as f:
        for u in range(len(G)):
            for v in G.edge[u]:
                f.write(f"{u} {v}\n")