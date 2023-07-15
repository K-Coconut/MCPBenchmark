import time
import os
import pickle
import argparse
from collections import defaultdict
import tracemalloc

import networkx as nx
import numpy as np
import torch

from environment import TestEnv
import utils


def evaluate(dqn, env, budgets, n_iter, action_factor, num_rr_set, result_dir, logger):

    for k in budgets:
        num_nodes = []
        num_edges = []
        for episode in range(n_iter):
            result_file = os.path.join(result_dir, f"budget{k}_factor_{action_factor}_iter_{episode}.txt")
            #if os.path.exists(result_file):
            #    continue
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
                next_state, done = env.step(action)
                state = next_state
            
            explore_subgraph_time = time.time() - st
            logger.info(f"explore time {explore_subgraph_time}")
            coverage, heuristic_runtime, seeds = env.run_imm_on_final_subgraph(k, num_rr_set)
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"episode {episode + 1}: k: {k}\tcoverage: {coverage}\ttotal runtime: {explore_subgraph_time + heuristic_runtime}\tCurrent memory: {current / 10 ** 6:.3f}\tPeak memory: {peak / 10 ** 6:.3f}")
            with open(result_file, "w") as f:
                f.write(f"{coverage} {explore_subgraph_time + heuristic_runtime} {peak}")
            
            num_nodes.append(env.state.number_of_nodes())
            num_edges.append(env.state.number_of_edges())
        logger.info(f"Budget {k}, Final graph size: N={np.mean(num_nodes):.3f} M={np.mean(num_edges):.3f}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("-m", '--weight_model', default='TV', help='weight model')
    args.add_argument("--gpu", type=int, default=0)
    args.add_argument("-n", "--n_iter", type=int, default=1)
    args.add_argument("--chunksize", type=int, default=28)
    args.add_argument("--subgraph_size", type=int, default=750)
    args.add_argument("--soln_budget", type=int, default=100)
    args.add_argument("--num_rr_set", type=int, default=100000)
    args.add_argument("-f", "--action_factor", type=int, default=5)
    args.add_argument('--point_proportion', default=20, type=int, help="point proportion that training dataset takes up")
    args.add_argument('--multi_epoch', action='store_true', help="tested with trained ckpt of different epoch")
    opt = args.parse_args()
    
    train_size_proportion = opt.point_proportion
    if train_size_proportion != 20:
        log_path = f"dqn_test_trainsize_{train_size_proportion}_{opt.dataset}_{opt.weight_model}.log"
    else:
        log_path = f"dqn_test_{opt.dataset}_{opt.weight_model}.log"
    logger = utils.get_logger("dqn_test", os.path.join("log", log_path))
    logger.info("=" * 150)
    logger.info(opt)
    
    utils.set_seed(seed=1)
    
    budgets = [20, 50, 100, 150, 200]
    num_rr_set = opt.num_rr_set
    encoder_name = "encoder"
    n_iter = opt.n_iter
    chunksize = opt.chunksize
    soln_budget = opt.soln_budget
    subgraph_size = opt.subgraph_size
    action_factor = opt.action_factor
    device = device = f"cuda:{opt.gpu}" if opt.gpu else "cpu"
    # fix youtube trained model
    if train_size_proportion != 20:
        budgets = [20]
        trained_model_prefix = os.path.join("youtube", opt.weight_model, "train_size_exp", f"train_size_{train_size_proportion}", "train")
        graph_name = os.path.join(
            opt.dataset, opt.weight_model, "train_size_exp", f"train_size_{train_size_proportion}",
        )
        test_graph_name = os.path.join(graph_name, "test")
    else:
        if opt.weight_model == "LND":
            trained_model = opt.dataset
        else:
            trained_model = "youtube"
        trained_model_prefix = os.path.join(trained_model, opt.weight_model, "train")
        graph_name = os.path.join(opt.dataset, opt.weight_model)
        test_graph_name = os.path.join(graph_name, "test")
    
    result_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/"
    os.makedirs(result_dir, exist_ok=True)
    
    tracemalloc.start()
    encoder = torch.load(f"data/{trained_model_prefix}/budget_{soln_budget}/{encoder_name}/{encoder_name}.pth", map_location=torch.device(device))    
    best_embeddings = None
    encoder.to("cpu")
    
    logger.info(f"reading edge file: {graph_name}/edges.txt")
    graph = nx.read_edgelist(os.path.join("data", graph_name, "test", "edges.txt"), nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    logger.info(f"graph size: N={graph.number_of_nodes()} M={graph.number_of_edges()}")    
    logger.info("building test environment")
    st = time.time()
    env = TestEnv(graph, soln_budget, subgraph_size, encoder, test_graph_name, chunksize=chunksize, beta=50, device=device)
    logger.info(f"build test environment takes {time.time() - st:.3f} seconds")
    
    if not opt.multi_epoch: 
        with open(f"data/{trained_model_prefix}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="rb") as f:
            dqn = pickle.load(f)
        dqn.epsilon = 0.01
        evaluate(dqn, env, budgets, n_iter, action_factor, num_rr_set, result_dir, logger)
    else:
        budgets = [20]
        num_epochs = 300
        for epoch in range(1, num_epochs + 1):
            result_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/train_epoch_{epoch}/"
            os.makedirs(result_dir, exist_ok=True)
            logger.info(f"using trained-epoch {epoch} model")
            with open(f"data/{trained_model_prefix}/budget_{soln_budget}/{encoder_name}/ckpt/trained_dqn_{epoch}", mode="rb") as f:
                dqn = pickle.load(f)
            evaluate(dqn, env, budgets, n_iter, action_factor, num_rr_set, result_dir, logger)
    tracemalloc.stop()
    tracemalloc.clear_traces()
