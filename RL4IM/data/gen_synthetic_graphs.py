import networkx as nx
import numpy as np
import os
import random

data_dir = 'synthetic/G%d/'


m = 3
p = 0.05

# n = 200, weight = 0.1
seed = 0
random.seed(seed)
np.random.seed(seed)

rg = (0, 10)

for i in range(*rg):
    ouput = data_dir % i
    os.makedirs(ouput, exist_ok=True)
    n = 200
    n += np.random.choice(range(-20, 21, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        weight = 0.1
        f.write('%d %d %f\n' % (edge[0], edge[1], weight))


# n = 200, weight = 1 / in_degree
seed = 0
random.seed(seed)
np.random.seed(seed)

rg = (10, 20)

for i in range(*rg):
    ouput = data_dir % i
    os.makedirs(ouput, exist_ok=True)
    n = 200
    n += np.random.choice(range(-20, 21, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        weight = 1 / g.in_degree(edge[1])
        f.write('%d %d %f\n' % (edge[0], edge[1], weight))


# n = 2000, weight = 0.1
seed = 0
random.seed(seed)
np.random.seed(seed)

rg = (20, 30)

for i in range(*rg):
    ouput = data_dir % i
    os.makedirs(ouput, exist_ok=True)
    n = 2000
    n += np.random.choice(range(-200, 200, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        weight = 0.1
        f.write('%d %d %f\n' % (edge[0], edge[1], weight))


# n = 2000, weight = 1 / in_degree
seed = 0
random.seed(seed)
np.random.seed(seed)
rg = (30, 40)

for i in range(*rg):
    ouput = data_dir % i
    os.makedirs(ouput, exist_ok=True)
    n = 2000
    n += np.random.choice(range(-200, 200, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        weight = 1 / g.in_degree(edge[1])
        f.write('%d %d %f\n' % (edge[0], edge[1], weight))


# n=20000, weight=0.1
seed = 0
random.seed(seed)
np.random.seed(seed)
rg = (40, 50)

for i in range(*rg):
    ouput = data_dir % i
    os.makedirs(ouput, exist_ok=True)
    n = 20000
    n += np.random.choice(range(-2000, 2000, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        weight = 0.1
        f.write('%d %d %f\n' % (edge[0], edge[1], weight))

# n=20000, weight=1 / in_degree
seed = 0
random.seed(seed)
np.random.seed(seed)
rg = (50, 60)

for i in range(*rg):
    ouput = data_dir % i
    os.makedirs(ouput, exist_ok=True)
    n = 20000
    n += np.random.choice(range(-2000, 2000, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        weight = 1 / g.in_degree(edge[1])
        f.write('%d %d %f\n' % (edge[0], edge[1], weight))


# generate validation graphs
seed = 100000
random.seed(seed)
np.random.seed(seed)

rg = (0, 5)

for i in range(*rg):
    ouput = "synthetic/validation/G%d/" % i
    os.makedirs(ouput, exist_ok=True)
    n = 200
    n += np.random.choice(range(-20, 21, 1))
    g = nx.powerlaw_cluster_graph(n=n, m=m, p=p, seed=seed + i)
    g = g.to_directed()
    f = open(ouput + 'attribute.txt', 'w')
    f.write("n=%d\nm=%d" % (g.number_of_nodes(), g.number_of_edges()))
    f = open(ouput + 'edges.txt', 'w')
    for edge in g.edges():
        f.write('%d %d %f\n' % (edge[0], edge[1], 1 / g.in_degree(edge[1])))
