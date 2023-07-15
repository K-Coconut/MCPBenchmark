import time
import os
import pickle
import argparse
import tracemalloc

import networkx as nx
import torch

from environment import TestEnv
import utils

from dqn_test import evaluate


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("--gpu", type=int, default=0)
    args.add_argument("-n", "--n_iter", type=int, default=1)
    args.add_argument("--chunksize", type=int, default=28)
    args.add_argument("--subgraph_size", type=int, default=750)
    args.add_argument("--soln_budget", type=int, default=100)
    args.add_argument("--num_rr_set", type=int, default=100000)
    args.add_argument("-f", "--action_factor", type=int, default=5)
    opt = args.parse_args()
    

    log_path = f"dqn_test_{opt.dataset}_CONST.log"
    logger = utils.get_logger("dqn_test_with_different_weight_model", os.path.join("log", log_path))
    logger.info("=" * 150)
    logger.info(opt)
    
    utils.set_seed(seed=1)
    
    budgets = [20]
    num_rr_set = opt.num_rr_set
    encoder_name = "encoder"
    n_iter = opt.n_iter
    chunksize = opt.chunksize
    soln_budget = opt.soln_budget
    subgraph_size = opt.subgraph_size
    action_factor = opt.action_factor
    device = device = f"cuda:{opt.gpu}" if opt.gpu else "cpu"
    
    fixed_weight_model = "CONST"
    trained_model_prefix = os.path.join("youtube", fixed_weight_model, "train")
    encoder = torch.load(f"data/{trained_model_prefix}/budget_{soln_budget}/{encoder_name}/{encoder_name}.pth", map_location=torch.device(device))    
    best_embeddings = None
    encoder.to("cpu")
    with open(f"data/{trained_model_prefix}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="rb") as f:
        dqn = pickle.load(f)
        dqn.epsilon = 0.01
    
    for weight_model in ["WC", "TV"]:
        tracemalloc.start()
        logger.info(f"Testing on {weight_model} with CONST model")
        graph_name = os.path.join(opt.dataset, weight_model)
        test_graph_name = os.path.join(graph_name, "test")
        
        result_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/test_with_CONST_model/"
        os.makedirs(result_dir, exist_ok=True)
        
        logger.info(f"reading edge file: {graph_name}/edges.txt")
        graph = nx.read_edgelist(os.path.join("data", graph_name, "test", "edges.txt"), nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        logger.info(f"graph size: N={graph.number_of_nodes()} M={graph.number_of_edges()}")    
        logger.info("building test environment")
        st = time.time()
        
        env = TestEnv(graph, soln_budget, subgraph_size, encoder, test_graph_name, chunksize=chunksize, beta=50, device=device)
        logger.info(f"build test environment takes {time.time() - st:.3f} seconds")
        evaluate(dqn, env, budgets, n_iter, action_factor, num_rr_set, result_dir, logger)
        
        tracemalloc.stop()
        tracemalloc.clear_traces()
        
        new_result = float(open(os.path.join(result_dir, f"budget20_factor_{action_factor}_iter_0.txt")).read().split()[0])
        original_result = float(open(os.path.join(result_dir, "..", f"budget20_factor_{action_factor}_iter_0.txt")).read().split()[0])
        logger.info(f"original result: {original_result:.3f}\tnew result: {new_result:.3f}\tgap: {(original_result - new_result) / original_result * 100:.2f}")