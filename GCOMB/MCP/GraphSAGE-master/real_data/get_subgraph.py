import sys
import random
import os

dataset = sys.argv[1]
os.system("mkdir -p " + dataset + "/")
os.system("mkdir -p " + dataset + "/train")
os.system("mkdir -p " + dataset + "/test")

SEED = 0
random.seed(SEED)

DATA_ROOT = f"../../../../data/{dataset}/"
full_graph_edge_file_path = DATA_ROOT + "/edges.txt"
train_sub_graph_edge_file_path = dataset + "/train/edges.txt"
test_sub_graph_edge_file_path = dataset + "/test/edges.txt"

train_point = 10

f_full_graph_edge = open(full_graph_edge_file_path, 'r')
f_train_sub_graph_edge = open(train_sub_graph_edge_file_path, 'w')

counter = 0

node_dic = {}
sep = '\t'

while True:
    line = f_full_graph_edge.readline()
    if not line:
        break

    if line[0] == "#":
        continue

    edge = line.replace('\n', '').split(sep)
    if len(edge) == 1:
        sep = ' '
        edge = line.replace('\n', '').split(sep)
    n1, n2 = edge[0], edge[1]
    random_int = random.randint(0, 100)
    if n1 not in node_dic:
        node_dic[n1] = len(node_dic)
    if n2 not in node_dic:
        node_dic[n2] = len(node_dic)
    if random_int < train_point:
        f_train_sub_graph_edge.write(f'{node_dic[n1]} {node_dic[n2]}\n')

f_train_sub_graph_edge.close()
