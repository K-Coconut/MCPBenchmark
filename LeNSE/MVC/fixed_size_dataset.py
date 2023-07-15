import networkx as nx
import random
import numpy as np
from functions import make_graph_features_for_encoder, close_pool, get_fixed_size_subgraphs, cover, greedy_mvc
import time
import pickle
import argparse
import glob
import os
import utils


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', default="BrightKite")
    args.add_argument("-k", '--budget', default=100)
    args.add_argument('--num_samples', default=100, type=int)
    args.add_argument('--fixed_size', default=200, type=int)
    opt = args.parse_args()
    
    random.seed(1)
    np.random.seed(1)
    NUM_SAMPLES = opt.num_samples
    NUM_CHECKPOINTS = 1
    BUDGET = opt.budget
    FIXED_SIZE = opt.fixed_size
    dataset = opt.dataset
    
    graph_dir = f"data/{dataset}/train"
    os.makedirs("log", exist_ok=True)
    logger = utils.get_logger("fixed_size_dataset", 
                              os.path.join("log", f"fixed_size_dataset_{dataset}.log"))
    logger.info("=" * 150)
    logger.info(opt)
    
    graph = nx.read_edgelist(os.path.join(graph_dir, "edges.txt"), nodetype=int, create_using=nx.Graph())
    with open(f"{graph_dir}/budget_{BUDGET}/score_and_seeds", mode="rb") as f:
        good_seeds, best_score = pickle.load(f)

    logger.info(f"size of all good seeds candidate: {len(good_seeds)}")
    logger.info(f"best score: {best_score}")
    logger.info(f"size of graph: N={graph.number_of_nodes()}, M={graph.number_of_edges()}")
    start = time.time()
    graph_features = make_graph_features_for_encoder(graph, graph_dir)
    N_PER_LOOP = NUM_SAMPLES // NUM_CHECKPOINTS
    count = 0
    for i in range(NUM_CHECKPOINTS):
        subgraphs_partion = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, logger=logger)
        for label, subgraphs in subgraphs_partion.items():
            with open(f"{graph_dir}/budget_{BUDGET}/data_{label}", mode="wb") as f:
                pickle.dump(subgraphs, f)
        del subgraphs_partion
    
    end = time.time()
    logger.info(f"time elapased: {end - start} seconds")
    
    subgraphs = []
    for fname in glob.glob(f"{graph_dir}/budget_{BUDGET}/data_*"):
        with open(fname, mode="rb") as f:
            hold = pickle.load(f)
            subgraphs += hold

    with open(f"{graph_dir}/budget_{BUDGET}/graph_data", mode="wb") as f:
        pickle.dump(subgraphs, f)

    for fname in glob.glob(f"{graph_dir}/budget_{BUDGET}/data_*"):
        os.remove(fname)
