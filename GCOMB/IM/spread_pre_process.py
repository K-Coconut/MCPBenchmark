import sys
import random
from functools import partial
import networkx as nx
import json
import os

import multiprocessing as mp


def create_num_MC_sim_copies(main_graph, graph_dir, mc_iter):
    print("mc iter", mc_iter)
    print(" len of graph", main_graph.number_of_nodes())
    print("graph_dir", graph_dir)
    for edge in main_graph.edges():
        inf_prob = main_graph.get_edge_data(*edge)['weight']
        sample_prob = random.uniform(0.0, 1.0)
        if sample_prob >= inf_prob:
            main_graph.remove_edge(*edge)

    graph_name = graph_dir + '/' + str(mc_iter) + "-G.txt"
    with open(graph_name, 'w') as f:
        for edge in main_graph.edges():
            f.write('%d %d\n' % edge)
    print(" written graph  to ", graph_name)
    return mc_iter


def mp_pool_format(G, graph_dir, mc_iter):
    return create_num_MC_sim_copies(G.copy(), graph_dir, mc_iter)


if __name__ == '__main__':

    dataset = sys.argv[1]
    model = sys.argv[2]
    graph_dir = os.path.join("GraphSAGE-master", "real_data", dataset, model, "train")
    NUM_MC_SIM = 10

    SEED = 0
    random.seed(SEED)

    pool = mp.Pool(processes=mp.cpu_count())

    mc_sim_directory_path = os.path.join(graph_dir, "mc_sim_graphs")
    print("mc_sim_dir pat", mc_sim_directory_path)
    os.makedirs(mc_sim_directory_path, exist_ok=True)
    G = nx.read_edgelist(os.path.join(graph_dir, "edges.txt"), data=(('weight', float),), nodetype=int,
                         create_using=nx.DiGraph())
    print(" graph loaded")

    pool_args = partial(mp_pool_format, G, mc_sim_directory_path)
    mc_sim = [x for x in range(0, NUM_MC_SIM)]
    print(" starting work")
    for iter, res in enumerate(pool.imap_unordered(pool_args, mc_sim, chunksize=2)):
        print("iter", res)

