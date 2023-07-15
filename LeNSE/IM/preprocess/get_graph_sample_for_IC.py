import os
import argparse
import multiprocessing as mp
import numpy as np
import networkx as nx



def make_subgraph(count, edges_, p_, path, weight_model):
    outcomes = np.random.binomial(1, p=1 - p_)
    G = nx.DiGraph()
    edges_to_keep = list(map(tuple, edges_[outcomes == 0]))
    G.add_edges_from(edges_to_keep)
    
    with open(os.path.join(path, f"subgraph_{count}.txt"), 'w') as f:
        for edge in G.edges():
            u, v = edge[0], edge[1]
            if weight_model == 'TV':
                weight = np.random.choice([0.1, 0.01, 0.001])
            elif weight_model == 'WC':
                weight = 1 / G.in_degree(v)
            elif weight_model == 'CONST':
                weight = 0.1
            elif weight_model == 'LND':
                weight = G[u][v]['weight']
            else:
                assert False, f"Wrong model: {weight_model}"
            G[u][v]['weight'] = weight
            f.write(str(u) + ' ' + str(v) + ' ' + str(weight) + '\n')
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("-m", '--weight_model', default='TV', help='weight model')
    args.add_argument("-t", '--mode', default='train', help='train or test')
    args.add_argument('--num_samples', default=1000, help='number of subgraphs to generate')
    args.add_argument('--chunksize', default=80)
    opt = args.parse_args()
    print(opt)
    
    SEED = 1
    np.random.seed(SEED)
    
    dataset = opt.dataset
    weight_model = opt.weight_model
    mode = opt.mode
    graph_path_dir = os.path.join(os.path.dirname(__file__), "..", "data", dataset, weight_model, mode)
    dataset_path = os.path.join(graph_path_dir, "edges.txt")
    graph = nx.read_edgelist(dataset_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    subgraph_path_dir = os.path.join(graph_path_dir, 'subgraph')
    os.makedirs(subgraph_path_dir, exist_ok=True)
    
    
    counts = [i for i in range(1, opt.num_samples + 1)]
    p = [graph[u][v]['weight'] for u, v in graph.edges()]
    p = np.array(p)
    p = [p for i in range(len(counts))]
    edges = np.array(graph.edges())
    edges = [edges for _ in range(len(counts))]
    subgraph_path_dir = [subgraph_path_dir for _ in range(len(counts))]
    weight_model = [weight_model for _ in range(len(counts))]
    del graph
    
    with mp.Pool() as pool:
        pool.starmap(make_subgraph, zip(counts, edges, p, subgraph_path_dir, weight_model), chunksize=opt.chunksize)
    
    
    