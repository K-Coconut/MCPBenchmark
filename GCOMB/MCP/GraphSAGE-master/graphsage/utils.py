from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
#assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

def load_data(prefix, normalize=True, load_walks=False, test_mode = False):
	G = nx.read_edgelist(prefix + "-edges.txt", nodetype=int)
	# sort nodes
	H = nx.Graph()
	H.add_nodes_from(sorted(G.nodes(data=True)))
	H.add_edges_from(G.edges(data=True))
	G = H
	if isinstance(G.nodes()[0], int):
		conversion = lambda n : int(n)
	else:
		conversion = lambda n : n
	id_map = json.load(open(prefix + "-id_map.json"))
	id_map = {conversion(k): int(v) for k, v in id_map.items()}
	walks = []
	print(" num nodes ", len(G))
	class_map = json.load(open(prefix + "-class_map.json"))
	if isinstance(list(class_map.values())[0], list):
		lab_conversion = lambda n: n
	else:
		lab_conversion = lambda n: int(n)

	class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}
	return G, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
	pairs = []
	for count, node in enumerate(nodes):
		if G.degree(node) == 0:
			continue
		for i in range(num_walks):
			curr_node = node
			for j in range(WALK_LEN):
				if len(G.neighbors(curr_node))==0:
					continue
				next_node = random.choice(G.neighbors(curr_node))
				# self co-occurrences are useless
				if curr_node != node:
					pairs.append((node,curr_node))
				curr_node = next_node
		if count % 1000 == 0:
			print("Done walks for", count, "nodes")
	return pairs

if __name__ == "__main__":
	""" Run random walks """
	graph_file = sys.argv[1]
	out_file = sys.argv[2]
	G_data = json.load(open(graph_file))
	G = json_graph.node_link_graph(G_data)
	nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
	G = G.output_graph(nodes)
	pairs = run_random_walks(G, nodes)
	with open(out_file, "w") as fp:
		fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
