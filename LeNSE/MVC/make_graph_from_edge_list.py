import networkx as nx
from functions import relabel_graph
import numpy as np

np.random.seed(1)
graph_name = "gowalla"

f = open(f"../data/{graph_name}/edges.txt", mode="r")
lines = f.readlines()
edges = []
for line in lines:
    line = line.split()
    edges.append([int(line[0]), int(line[1])])

graph = nx.Graph()
graph.add_edges_from(edges)

graph, trans = relabel_graph(graph, True)

nx.write_gpickle(graph, f"data/{graph_name}/main")
