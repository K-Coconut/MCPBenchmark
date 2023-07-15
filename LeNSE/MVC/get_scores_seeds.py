import networkx as nx
import random
import numpy as np
from functions import calculate_cover, greedy_mvc, make_graph_features_for_encoder
import time
import pickle
import os
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", default="BrightKite")
    opt = args.parse_args()
    random.seed(1)
    np.random.seed(1)
    BUDGET = 100
    graph_name = opt.dataset
    print(graph_name)

    graph_dir = f"data/{graph_name}/train"
    graph = nx.read_edgelist(os.path.join(graph_dir, "edges.txt"), nodetype=int, create_using=nx.Graph())
    all_seeds = set()
    scores = []
    start = time.time()
    good_seeds = greedy_mvc(graph, BUDGET)
    best_score = calculate_cover(graph, good_seeds)
    end = time.time()
    print(f"It took {(end - start) / 60:.3f} minutes\n")

    # graph_features = make_graph_features_for_encoder(graph, graph_name)

    os.makedirs(f"{graph_dir}/budget_{BUDGET}/", exist_ok=True)

    with open(f"{graph_dir}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
        pickle.dump((good_seeds, best_score), f)

    with open(f"{graph_dir}/budget_{BUDGET}/time_taken_to_get_seeds", mode='w') as f:
        f.write(f"It took {(end - start) / 60} minutes to get a solution.")
