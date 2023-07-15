import os
import sys
import argparse
import pickle
import random
import time
import numpy as np
import networkx as nx

sys.path.append("..")
from functions import make_graph_features_for_encoder, call_imm
import utils

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("-m", '--weight_model', default='TV', help='weight model')
    args.add_argument("-t", '--mode', default='train', help='train or test')
    args.add_argument("-k", '--budget', default=100)
    args.add_argument('--point_proportion', default=20, type=int)
    opt = args.parse_args()

    # the author set this as 1.
    NUM_REPEAT_ITERATIONS = 10

    logger = utils.get_logger(
        "get_scores_seeds_counts",
        os.path.join(
            "log", f"get_scores_seeds_counts_{opt.dataset}_{opt.weight_model}_{opt.mode}.log"
        ),
    )
    logger.info(opt)
    logger.info("NUM_REPEAT_ITERATIONS: " + str(NUM_REPEAT_ITERATIONS))

    BUDGET = opt.budget
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)

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
    graph = nx.read_edgelist(
        dataset_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph()
    )

    counts = {}
    all_seeds = set()
    scores = []
    start = time.time()
    for i in range(NUM_REPEAT_ITERATIONS):
        logger.info(f"iter: {i}")
        seeds = call_imm(graph_path_dir, graph, BUDGET)

        for seed in seeds:
            if seed in counts:
                counts[seed] += 1
            else:
                counts[seed] = 1
        spread = utils.calculate_influence_spread(os.path.join(graph_path_dir, "edges.txt"), seeds)
        scores.append(spread)
        logger.info(f"spread: {spread}\nseeds: {str(seed)}")
        all_seeds |= set(seeds)
    good_seeds = all_seeds.copy()
    best_score = max(scores)
    end = time.time()
    logger.info(f"final good seeds size: {len(good_seeds)}\tbest score: {max(scores)}")
    logger.info(f"It took {(end - start) :.3f} seconds\n")

    graph_features = make_graph_features_for_encoder(graph, graph_path_dir)

    os.makedirs(f"{graph_path_dir}/budget_{BUDGET}/", exist_ok=True)

    with open(f"{graph_path_dir}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
        pickle.dump((good_seeds, best_score, counts), f)

    with open(f"{graph_path_dir}/budget_{BUDGET}/time_get_seeds", mode='w') as f:
        f.write(f"{end - start}")
