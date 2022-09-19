from pathlib import Path
import os
import sys
import time
import re
import argparse
from copy import deepcopy
from types import SimpleNamespace

import yaml
import networkx as nx
import numpy as np

from src.runner import runners
from src.agent.rl4im.agent import DQAgent
from src.agent.baseline import *
from src.environment.env import Environment
from src.environment.graph import Graph
from src.runner.utils import load_checkpint
from src.utils.load_config import get_latest_checkpoint_path, get_config
from src.utils.logging import get_logger, Logger

from main import _get_basic_config
from change import runIC_repeat
from main import recursive_dict_update


epochs = range(0, 20000, 1000)
graph_size = [80, 100, 200, 300, 500, 700, 1000]

size_to_checkpoint = {80: 47, 100: 46, 200: 42, 300: 45}


def _load_graph(args):
    graph_dic = {}

    for i, graph_ in enumerate(range(args.graph_nbr_train, args.graph_nbr_train + args.graph_nbr_test)):
        seed = 200000 + i
        g = nx.read_edgelist(os.path.join(args.graph_path, 'G%d' % graph_, 'edges.txt'), nodetype=int,
                             data=(("weight", float),), create_using=nx.DiGraph)
        graph_dic[graph_] = Graph(g=g, graph_type=args.graph_type, cur_n=args.node_test, p=args.p, m=args.m, seed=seed,
                                  args=args, is_train=False)
        graph_dic[graph_].graph_name = str(seed)
    graph_dic[0] = graph_dic[args.graph_nbr_train]

    return graph_dic


def evaluate(args):
    graph_dic = _load_graph(args)

    np.random.seed(args.seed)
    env_class = Environment(cascade=args.cascade, T=args.T, budget=args.budget, q=args.q, graphs=graph_dic, args=args)
    agent = DQAgent(graph_dic, args.model, args.lr, args.bs, args.n_step, args=args)
    if args.use_cuda:
        agent.cuda()
    logger = Logger(get_logger())
    my_runner = runners.Runner(args, env_class, agent, args.verbose, logger=logger)
    load_checkpint(args=args, runner=my_runner, agent=agent)
    my_runner.evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", default='basic_env')
    parser.add_argument("--config", default='rl4im')
    parser.add_argument("--model_prefix", type=str, default="sacred", help="sacred, sample_size_sacred, graph_size_sacred")
    parser.add_argument("-p", "--checkpoint_path", default=None, type=int)
    argv = parser.parse_args()
    params = deepcopy(sys.argv)
    params_bak = deepcopy(sys.argv)

    # Get the defaults from task_default.yaml
    with open(os.path.join(os.path.dirname(__file__), "src", "tasks", "config", "task_default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "task_default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = get_config(argv.env_config)
    alg_config = get_config(argv.config)
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict['is_real_graph'] = False  # synthetic graphs
    config_dict['mode'] = 'test'
    task_name = argv.config

    config_dict['task_name'] = task_name
    config_dict['seed'] = 0
    MODEL_DIR = "models/"
    config_dict['graph_path'] = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))), 'data', 'synthetic', 'validation')
    
    argv.checkpoint_path = argv.checkpoint_path
    
    config_dict['checkpoint_path'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  f"{MODEL_DIR}/{argv.model_prefix}/{argv.checkpoint_path}/models/")

    trained_epochs = sorted(
        [int(re.match("\d+", i).group(0)) for i in os.listdir(config_dict['checkpoint_path']) if re.match("\d+", i)])

    VALIDATION_BUDGET = 20
    NUM_VALIDATION_GRAPHS = 5
    config_dict['graph_nbr_train'] = 0
    config_dict['graph_nbr_test'] = NUM_VALIDATION_GRAPHS
    config_dict['budget'] = VALIDATION_BUDGET
    config_dict['T'] = VALIDATION_BUDGET

    for epoch in trained_epochs:
        config_dict['load_step'] = epoch
        save_path = os.path.join(config_dict['graph_path'], 'G%d', 'epoch', str(epoch))
        config_dict['local_results_path'] = save_path
        args = SimpleNamespace(**config_dict)

        print("=" * 50, "budget: %d" % VALIDATION_BUDGET, "=" * 50)
        evaluate(args)

    # select the best
    print("*" * 20, "finding the best epoch model", "*" * 20)
    args = SimpleNamespace(**config_dict)
    graph_dict = _load_graph(args)

    epoch_rewards = np.empty((len(trained_epochs), NUM_VALIDATION_GRAPHS))
    best_reward_so_far = 0.
    best_epoch_so_far = trained_epochs[0]
    os.makedirs(os.path.join(config_dict['checkpoint_path'], "model_log"), exist_ok=True)
    for i, epoch in enumerate(trained_epochs):
        for g_index in range(0, NUM_VALIDATION_GRAPHS):
            seedfile = os.path.join(config_dict['graph_path'], f"G{g_index}/epoch/{epoch}", f'budget{VALIDATION_BUDGET}.txt')
            seeds = [int(i) for i in open(seedfile).read().split("\n") if i]
            G = graph_dict[g_index].g
            reward = runIC_repeat(G, seeds, sample=1000)[0]
            print(f"epoch {epoch}, G{g_index}: {reward}")
            epoch_rewards[i, g_index] = reward
        if np.mean(epoch_rewards[i]) > best_reward_so_far:
            best_reward_so_far = np.mean(epoch_rewards[i])
            best_epoch_so_far = epoch
        with open(os.path.join(config_dict['checkpoint_path'], "model_log", f"best_model_epoch_{epoch}.txt"), 'w') as f:
            f.write(str(best_epoch_so_far))

    average_reward = np.mean(epoch_rewards, axis=1)
    best_epoch = trained_epochs[np.argmax(average_reward)]
    print(f"average_reward: {average_reward}")
    print(f"best epoch: {best_epoch}")
    np.save(os.path.join(config_dict['checkpoint_path'], 'epoch_rewards'), epoch_rewards)
    with open(os.path.join(config_dict['checkpoint_path'], 'best_epoch.txt'), 'w') as f:
        f.write(f"{best_epoch}")
