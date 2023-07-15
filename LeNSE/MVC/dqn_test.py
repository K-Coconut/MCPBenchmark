import os
import argparse
import time
import pickle
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch

import utils
from environment import TestEnv, BigGraph
from functions import relabel_graph


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("--gpu", type=int, default=0)
    args.add_argument("-n", "--n_iter", type=int, default=1)
    args.add_argument("--subgraph_size", type=int, default=750)
    args.add_argument("--soln_budget", type=int, default=100)
    args.add_argument("-f", "--action_factor", type=int, default=5)
    opt = args.parse_args()
    
    logger = utils.get_logger("dqn_test", os.path.join("log", f"dqn_test_{opt.dataset}.log"))
    logger.info("=" * 150)
    logger.info(opt)
    
    utils.set_seed(1)
    budgets = [20, 50, 100, 150, 200]
    encoder_name = "encoder"
    dataset = opt.dataset
    n_iter = opt.n_iter
    soln_budget = opt.soln_budget
    subgraph_size = opt.subgraph_size
    action_factor = opt.action_factor
    device = device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"
    
    # fix BrightKite trained model
    train_graph_name = os.path.join("BrightKite", "train")
    test_graph_name = os.path.join(dataset, "test")
    
    result_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/"
    os.makedirs(result_dir, exist_ok=True)

    tracemalloc.start()

    encoder = torch.load(f"data/{train_graph_name}/budget_{soln_budget}/{encoder_name}/{encoder_name}.pth", map_location=torch.device(device))
    best_embeddings = None
    encoder.to("cpu")
    
    logger.info(f"reading edge file: {test_graph_name}/edges.txt")
    graph = nx.read_edgelist(os.path.join("data", test_graph_name, "edges.txt"), nodetype=int, create_using=nx.Graph())
    graph = relabel_graph(graph)
    logger.info(f"graph size: N={graph.number_of_nodes()} M={graph.number_of_edges()}")    
    logger.info("building test environment")
    st = time.time()
    env = TestEnv(graph, soln_budget, subgraph_size, encoder, test_graph_name, beta=50, device=device)
    logger.info(f"build test environment takes {time.time() - st:.3f} seconds")
    
    
    with open(f"data/{train_graph_name}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="rb") as f:
        dqn = pickle.load(f)
        dqn.epsilon = 0.01    
        dqn.device = device
        dqn.net = dqn.net.to(device)
        dqn.net = dqn.net.eval()

    # env = TestEnv(graph, soln_budget, subgraph_size, encoder, test_graph_name, action_limit=action_limit, beta=50, cuda=cuda)
    env = BigGraph(graph, soln_budget, subgraph_size, encoder, test_graph_name, device=device)
    for k in budgets:
        num_nodes = []
        num_edges = []
        for episode in range(n_iter):
            result_file = os.path.join(result_dir, f"budget{k}_factor_{action_factor}_iter_{episode}.txt")
            count = 0
            if os.path.exists(result_file):
                continue
            utils.set_seed(seed=episode)
            logger.info(f"budget {k} starting episode {episode+1}")
            count = 0
            st = time.time()
            state = env.reset()
            env.action_limit = k * action_factor
            done = False
            while not done:
                count += 1
                action, state_for_buffer = dqn.act(state)
                next_state, reward, done = env.step(action)
                state = next_state

            num_nodes.append(env.state.number_of_nodes())
            num_edges.append(env.state.number_of_edges())
            
            explore_subgraph_time = time.time() - st
            logger.info(f"explore time {explore_subgraph_time}")
            cover, heuristic_runtime, seeds = env.calculate_cover_on_final_subgraph(budget=k)
            coverage = cover / graph.number_of_nodes()
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f'seeds: {seeds}')
            logger.info(f"episode {episode + 1}: k: {k}\tcoverage: {coverage}\ttotal runtime: {explore_subgraph_time + heuristic_runtime}\tCurrent memory: {current / 10 ** 6:.3f}MB\tPeak memory: {peak / 10 ** 6:.3f}MB")
            with open(result_file, "w") as f:
                f.write(f"{coverage} {explore_subgraph_time + heuristic_runtime} {peak}")
        logger.info(f"Budget {k}, Final graph size: N={np.mean(num_nodes):.3f} M={np.mean(num_edges):.3f}")