import os
import sys
import re
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


def _load_graph(args):
    graph_dic = {}
    for i, graph_ in enumerate(range(args.graph_nbr_train, args.graph_nbr_train + args.graph_nbr_test)):
        seed = 200000 + i
        g = nx.read_edgelist(os.path.join('data', 'synthetic', 'G%d' % graph_, 'edges.txt'), nodetype=int,
                             data=(("weight", float),), create_using=nx.DiGraph)
        graph_dic[graph_] = Graph(g=g, seed=seed, args=args, is_train=False)
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
    final_reward = my_runner.evaluate()
    print(final_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", default='basic_env')
    parser.add_argument("--config", default='rl4im')
    parser.add_argument("-p", "--checkpoint_path", default=None, type=int)
    parser.add_argument("--model_prefix", type=str, default="sacred",
                        help="sacred, sample_size_sacred, graph_size_sacred")
    parser.add_argument("--graph_load_start", type=int, default=0)
    parser.add_argument("--graph_load_end", type=int, default=60)
    argv = parser.parse_args()
    params = deepcopy(sys.argv)
    params_bak = deepcopy(sys.argv)

    BUDGET = 20
    MODEL_DIR = "models/"

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
    config_dict['is_real_graph'] = False  # synthetic graphs
    config_dict['mode'] = 'test'
    config_dict['graph_nbr_train'] = argv.graph_load_start
    config_dict['graph_nbr_test'] = argv.graph_load_end

    config_dict['seed'] = 0
    if argv.checkpoint_path is None:
        ckpt_path = get_latest_checkpoint_path(MODEL_DIR)
        argv.checkpoint_path = ckpt_path
    config_dict['checkpoint_path'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  f"{MODEL_DIR}/{argv.model_prefix}/{argv.checkpoint_path}/models/")

    trained_epochs = sorted(
        [int(re.match("\d+", i).group(0)) for i in os.listdir(config_dict['checkpoint_path']) if re.match("\d+", i)])

    last_best_model_so_far = 0
    for epoch in trained_epochs:
        best_model_so_far = int(
            open(os.path.join(config_dict['checkpoint_path'], "model_log", f"best_model_epoch_{epoch}.txt")).read())
        if best_model_so_far == last_best_model_so_far:
            print(f"epoch {epoch} using the same best model {best_model_so_far} as the last epoch, skip")
            continue
        last_best_model_so_far = best_model_so_far
        config_dict['load_step'] = best_model_so_far
        print(f"epoch{epoch}, best model so far: {best_model_so_far}")

        save_path = os.path.join("data", 'synthetic', 'G%d', argv.model_prefix, str(argv.checkpoint_path), 'epoch',
                                 str(epoch))
        config_dict['local_results_path'] = save_path
        config_dict['budget'] = BUDGET
        config_dict['T'] = BUDGET
        args = SimpleNamespace(**config_dict)

        print("=" * 50, "budget: %d" % BUDGET, "=" * 50)
        evaluate(args)
