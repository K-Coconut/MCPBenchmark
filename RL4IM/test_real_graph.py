from pathlib import Path
import os
import sys
import argparse
from copy import deepcopy
from types import SimpleNamespace

import yaml
import numpy as np

from src.runner import runners
from src.agent.rl4im.agent import DQAgent
from src.agent.baseline import *
from src.environment.env import Environment
from src.environment.graph import Graph
from src.runner.utils import load_checkpint
from src.utils.load_config import get_latest_checkpoint_path, get_config
from src.utils.logging import get_logger, Logger

from main import recursive_dict_update

DATA_DIR = "../data/"
MODEL_DIR = "models/"


def _load_graph(args):
    path = Path(DATA_DIR, args.real_graph_name, "IM", args.weight_model)
    print('Loading graph: ', args.real_graph_name)
    G = nx.read_edgelist(os.path.join(path, 'edges.txt'), nodetype=int, data=(("weight", float),),
                         create_using=nx.DiGraph)
    print("n=%d\tm=%d" % (G.number_of_nodes(), G.number_of_edges()))
    G = Graph(g=G, args=args, seed=args.seed)
    G.graph_name = str(0)
    return {0: G}


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
    evaluate_start = time.time()
    my_runner.evaluate()
    evaluate_end = time.time()
    print('runtime in seconds is: ', evaluate_end - evaluate_start)
    with open("%s/time_budget%d.txt" % (args.local_results_path, args.budget), 'w') as f:
        f.write(str(evaluate_end - evaluate_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--dataset', required=True)
    parser.add_argument("--weight_model", required=True)
    parser.add_argument("--env_config", default='basic_env')
    parser.add_argument("--config", default='rl4im')
    parser.add_argument("--results_dir", default='result')
    parser.add_argument("-p", "--checkpoint_path", default=None)
    parser.add_argument("-t", "--step", type=int, default=0, help="time step of training model to load")
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
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict['weight_model'] = argv.weight_model
    config_dict['load_step'] = argv.step
    config_dict['real_graph_name'] = argv.dataset
    config_dict['is_real_graph'] = True
    config_dict['graph_nbr_train'] = 0
    config_dict['graph_nbr_test'] = 1
    config_dict['mode'] = 'test'
    task_name = argv.config
    save_path = os.path.join(argv.results_dir, argv.dataset, argv.weight_model)
    config_dict['task_name'] = task_name
    config_dict['local_results_path'] = save_path
    config_dict['seed'] = 0
    if argv.checkpoint_path is None:
        ckpt_path = get_latest_checkpoint_path(MODEL_DIR)
        argv.checkpoint_path = ckpt_path
    config_dict['checkpoint_path'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  f"{MODEL_DIR}/sacred/{argv.checkpoint_path}/models/")

    for budget in [20, 50, 100, 150, 200]:
        config_dict['budget'] = budget
        config_dict['T'] = budget
        args = SimpleNamespace(**config_dict)

        print("=" * 50, "budget: %d" % budget, "=" * 50)
        evaluate(args)
