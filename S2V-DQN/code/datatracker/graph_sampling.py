import networkx as nx
import numpy as np
from tqdm import tqdm


def sample(G, path, num=100):
    nodes = []
    degree_list = []
    for n, degree in G.degree_iter():
        nodes.append(n)
        degree_list.append(degree)
    new_edges_file = open(path, 'w')
    node_dict = {}
    seed_set = set()
    total_weight = sum(degree_list)
    while len(seed_set) < num:
        seed = np.random.choice(nodes, p=[d / total_weight for d in degree_list])
        seed_set.add(seed)

    m = 0
    for u, v in G.edges(seed_set):
        if u not in node_dict:
            node_dict[u] = len(node_dict)
        if v not in node_dict:
            node_dict[v] = len(node_dict)
        m += 1
        new_edges_file.write("%d\t%d\n" % (node_dict[u], node_dict[v]))
    new_edges_file.write("# n=%d\n# m=%d" % (len(node_dict), m))
    print("n=%d\nm=%d" % (len(node_dict), m))


def load_graph(path):
    G = nx.Graph()
    for line in tqdm(open(path)):
        u = int(line.split('\t')[0])
        v = int(line.split('\t')[1])
        G.add_edge(u, v)
    return G

