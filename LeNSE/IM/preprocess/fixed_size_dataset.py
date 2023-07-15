import sys
import argparse
import glob
import os
import pickle
import time

import networkx as nx
import random
import numpy as np

sys.path.append("..")
from functions import make_graph_features_for_encoder, get_fixed_size_subgraphs
import utils

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("-m", '--weight_model', default='TV', help='weight model')
    args.add_argument("-t", '--mode', default='train', help='train or test')
    args.add_argument("-k", '--budget', default=100)
    args.add_argument('--num_samples', default=100, type=int)
    args.add_argument('--fixed_size', default=200, type=int)
    args.add_argument('--point_proportion', default=20, type=int)
    opt = args.parse_args()
    
    logger = utils.get_logger("fixed_size_dataset", 
                              os.path.join("log", f"fixed_size_dataset_{opt.dataset}_{opt.weight_model}_{opt.mode}.log"))
    logger.info("=" * 150)
    logger.info(opt)
    random.seed(1)
    np.random.seed(1)
    NUM_SAMPLES = opt.num_samples
    NUM_CHECKPOINTS = 1
    budget = opt.budget
    FIXED_SIZE = opt.fixed_size

    dataset = opt.dataset
    weight_model = opt.weight_model
    mode = opt.mode
    train_size_proportion = opt.point_proportion
    if train_size_proportion != 20:
        graph_path_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", dataset, weight_model, "train_size_exp", f"train_size_{train_size_proportion}", mode
        )
    else:
         graph_path_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", dataset, weight_model, mode
        )
    dataset_path = os.path.join(graph_path_dir, "edges.txt")
    graph = nx.read_edgelist(dataset_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    
    with open(f"{graph_path_dir}/budget_{budget}/score_and_seeds", mode="rb") as f:
        # counts: The number of occurrences in the seed set
        good_seeds, best_score, counts = pickle.load(f)

    logger.info(f"size of all good seeds candidate: {len(good_seeds)}")
    logger.info(f"best score: {best_score}")
    logger.info(f"size of graph: N={graph.number_of_nodes()}, M={graph.number_of_edges()}")
    start = time.time()
    graph_features = make_graph_features_for_encoder(graph, graph_path_dir)
    N_PER_LOOP = NUM_SAMPLES // NUM_CHECKPOINTS

    for i in range(NUM_CHECKPOINTS):
        subgraphs_partion = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, counts, budget, FIXED_SIZE, graph_path_dir, best_score, graph_features, logger=logger)
        for label, subgraphs in subgraphs_partion.items():
            with open(f"{graph_path_dir}/budget_{budget}/data_{label}", mode="wb") as f:
                pickle.dump(subgraphs, f)
        del subgraphs_partion

    end = time.time()
    logger.info(f"time elapased: {end - start} seconds")
    
    subgraphs = []
    for fname in glob.glob(f"{graph_path_dir}/budget_{budget}/data_*"):
        with open(fname, mode="rb") as f:
            hold = pickle.load(f)
            subgraphs += hold

    with open(f"{graph_path_dir}/budget_{budget}/graph_data", mode="wb") as f:
        pickle.dump(subgraphs, f)

    for fname in glob.glob(f"{graph_path_dir}/budget_{budget}/data_*"):
        os.remove(fname)

