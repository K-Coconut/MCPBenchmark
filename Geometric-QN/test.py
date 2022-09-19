# GCNQ: Multi Q
import numpy as np
import networkx as nx
import random, io
from multiprocessing import cpu_count

from rl_alg.dqn import DQNTrainer
from expts.net_env import NetworkEnv

import torch

from ge.models.deepwalk import DeepWalk
import pickle, os, time
import gc
import logging, argparse
import tracemalloc


def arg_parse():
    parser = argparse.ArgumentParser(description='Influence Maxima Arguments')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, default='BrightKite')
    parser.add_argument('--weight_model', dest='weight_model', type=str, default='CONST')
    parser.add_argument('--logfile', dest='logfile', type=str, default='test.log',
                        help='Logging file')
    parser.add_argument('--logdir', dest='logdir', type=str, default=None,
                        help='Tensorboard LogDir')
    parser.add_argument('--log-level', dest='loglevel', type=int, default=2, choices=[1, 2],
                        help='Logging level')
    parser.add_argument('-k', dest='budget', type=int, default=20,
                        help='Number of queries for sampling')
    parser.add_argument('--seed', dest='seed', type=int, default=10, help='random seed')
    parser.add_argument('--extra-seeds', dest='extra_seeds', type=int, default=5,
                        help='Initial number of random seeds')
    parser.add_argument('--prop-prob', dest='prop_probab', type=float, default=0.1,
                        help='Propogation Probability for each Node')
    parser.add_argument('--cpu', dest='cpu', type=int, default=0,
                        help='Number of CPUs to use for influence sampling')
    parser.add_argument('--samples', dest='samples', type=int, default=100,
                        help='Number of samples in Influence Maximization')
    parser.add_argument('--opt', dest='obj', type=float, default=0,
                        help='Threshold for reward')
    parser.add_argument('--infl-budget', dest='ibudget', type=int, default=10,
                        help='Number of queries during influence(greedy steps)')
    parser.add_argument('--render', dest='render', type=int, default=0,
                        help='1 to Render graphs, 0 to not')
    parser.add_argument('--write', dest='write', type=int, default=1,
                        help='1 to write stats to tensorboard, 0 to not')

    parser.add_argument('--change-seeds', dest='changeSeeds', type=int, default=0,
                        help='1 to change seeds after each episode, 0 to not')
    parser.add_argument('--add-noise', dest='add_noise', type=int, default=1,
                        help='1 to add noise to action 0 to not')
    parser.add_argument('--sep-net', dest='sep_net', type=int, default=0,
                        help='Seperate network rep for actor and critic')

    parser.add_argument('--save-freq', dest='save_every', type=int, default=100,
                        help='Model save frequency')
    parser.add_argument('--n-iter', dest='n_iter', type=int, default=3,
                        help='repeated times of experiments')

    parser.add_argument('--eps', dest='num_ep', type=int, default=10000,
                        help='Number of Episodes')
    parser.add_argument('--buffer-size', dest='buff_size', type=int, default=4000,
                        help='Replay buffer Size')

    parser.add_argument('--gcn_layers', dest='gcn_layers', type=int, default=2,
                        help='No. of GN Layers before each pooling')
    parser.add_argument('--num_poolig', dest='num_pooling', type=int, default=1,
                        help='No.pooling layers')
    parser.add_argument('--assign_dim', dest='assign_dim', type=int, default=100,
                        help='pooling hidden dims 1')
    parser.add_argument('--assign_hidden_dim', dest='assign_hidden_dim', type=int, default=150,
                        help='pooling hidden dims 2')

    parser.add_argument('--actiondim', dest='action_dim', type=int, default=60,
                        help='Action(Node) Dimensions')
    parser.add_argument('--const_features', dest='const_features', type=int, default=1,
                        help='1 to have constant features')
    parser.add_argument('--inputdim', dest='input_dim', type=int, default=20,
                        help='Node features Dimensions')

    parser.add_argument('--step_reward', dest='nop_reward', type=float, default=0,
                        help='Reward for each step')
    parser.add_argument('--bad_reward', dest='bad_reward', type=float, default=0,
                        help='Reward for each step that is closer to active')
    parser.add_argument('--norm_reward', dest='norm_reward', type=int, default=0,
                        help='Normalize reward with opt')
    parser.add_argument('--max_reward', dest='max_reward', type=int, default=None,
                        help='Normalize reward with opt')
    parser.add_argument('--min_reward', dest='min_reward', type=int, default=None,
                        help='Normalize reward with opt')

    parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--eta', dest='eta', type=float, default=0.1,
                        help='Target network transfer rate')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Discount rate')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1,
                        help='Epsilon exploration')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100,
                        help='Gradient Update Batch Size')

    parser.add_argument('--use_cuda', dest='use_cuda', type=int, default=1,
                        help='1 to use cuda 0 to not')
    parser.add_argument('--walk_len', dest='walk_len', type=int, default=10,
                        help='Walk Length')

    parser.add_argument('--num_walks', dest='num_walks', type=int, default=80,
                        help='Walk Length')
    parser.add_argument('--win', dest='win', type=int, default=5,
                        help='Window size')
    parser.add_argument('--emb_iters', dest='emb_iters', type=int, default=50,
                        help='Walk Length')

    parser.add_argument('--noise_momentum', dest='noise_momentum', type=float, default=0.15,
                        help='Noise Momentum')
    parser.add_argument('--noise_magnitude', dest='noise_magnitude', type=float, default=0.2,
                        help='Noise Magnitude')

    parser.add_argument('--noise_decay', dest='noise_decay_rate', type=float, default=0.999,
                        help='Noise Decay Rate')
    parser.add_argument('--eta_decay', dest='eta_decay', type=float, default=1.,
                        help='eta Decay Rate')
    parser.add_argument('--alpha_decay', dest='alpha_decay', type=float, default=1.,
                        help='alpha Decay Rate')
    parser.add_argument('--eps_decay', dest='eps_decay_rate', type=float, default=0.999,
                        help='Epsilon Decay Rate')

    parser.add_argument('--sample_times', dest='times_mean', type=int, default=10,
                        help='Number of times to sample objective from fluence algorithm')
    parser.add_argument('--sample_times_env', dest='times_mean_env', type=int, default=5,
                        help='Number of times to sample objective from fluence algorithm for env rewards')
    parser.add_argument('--model_prefix', dest='model_prefix', type=str, default='1_trainset_',
                        help='Prefix of the saved model')
    parser.add_argument('--neigh', dest='k', type=int, default=1,
                        help='K nearest for ation')

    return parser.parse_args()


def load_checkpoint(model, model_prefix, path='models'):
    epochs = [int(i[len(model_prefix):-4]) for i in os.listdir(path) if i.startswith(model_prefix)]
    if not epochs:
        print("NO MODEL LOADED")
        return 0
    latest = max(epochs)
    model_path = '%s%d.pth' % (model_prefix, latest)
    print("Loading model: ", model_path)
    model.load_models(model_path)
    print("Model loaded")
    return latest


def make_const_attrs(graph, input_dim):
    n = len(graph)
    mat = np.ones((n, input_dim))
    # mat = np.random.rand(n,input_dim)
    return mat


def make_env_attrs_1(env, embs, n, input_dim):
    mat1 = np.zeros((n, int(action_dim + 2)))
    for u in env.active:
        mat1[u, :-2] = embs[u]
        mat1[u, -2] = 1
    for u in env.possible_actions:
        mat1[u, :-2] = embs[u]
        mat1[u, -1] = 1
    return mat1


def get_action_curr1(s, emb, nodes):
    q_vals = -10000.0
    node = -1
    for v in nodes:
        value, _ = acmodel.get_values2(s[0], s[1], emb[v])
        if value > q_vals:
            q_vals = value
            node = v
    return node, q_vals


def get_embeds(g):
    d = {}
    for n in g.nodes:
        d[n] = str(n)
    g1 = nx.relabel_nodes(g, d)
    graph_model = DeepWalk(g1, num_walks=args.num_walks, walk_length=args.walk_len,
                           workers=args.cpu if args.cpu > 0 else cpu_count())

    graph_model.train(window_size=args.win, iter=args.emb_iters, embed_size=action_dim)
    embs = {}
    emb1 = graph_model.get_embeddings()
    for n in emb1.keys():
        embs[int(n)] = emb1[n]

    return embs


args = arg_parse()
extra_seeds = args.extra_seeds

# n = 100
logfile = args.logfile

logging.basicConfig(level=args.loglevel * 10, filename=logfile, filemode='w', datefmt='%d-%b-%y %H:%M:%S',
                    format='%(levelname)s - %(asctime)s - %(message)s ')

budget = args.budget
render = args.render
write = args.write

debug = False

action_dim = args.action_dim
if args.const_features:
    input_dim = args.input_dim
else:
    input_dim = args.action_dim + 2
nop_reward = args.nop_reward

LR = args.lr
eta = args.eta
batch_size = args.batch_size
gcn_layers = args.gcn_layers
num_pooling = args.num_pooling
assign_dim = args.assign_dim
assign_hidden_dim = args.assign_hidden_dim

use_cuda = args.use_cuda

noise_momentum = args.noise_momentum
noise_magnitude = args.noise_magnitude

noise_decay_rate = args.noise_decay_rate
eta_decay = args.eta_decay

times_mean = args.times_mean
model_prefix = args.model_prefix

noise_param = 1

# load graph
dataset = args.dataset
DATA_DIR = '../data/'
g_path = os.path.join(DATA_DIR, dataset, "IM", args.weight_model, "edges.txt")

G = nx.read_edgelist(g_path, nodetype=int, data=(('p', float),), create_using=nx.DiGraph)
print(g_path)
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
logging.info("Nodes: " + str(G.number_of_nodes()) + ' Edges: ' + str(G.number_of_edges()))

logging.info('State Dimensions: ' + str(action_dim))
logging.info('Action Dimensions: ' + str(action_dim))

node_attrs = make_const_attrs(G, input_dim)

tracemalloc.start()

OUTPUT_DIR = os.path.join("result", dataset, args.weight_model, model_prefix)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(args.n_iter):
    if os.path.exists(os.path.join(OUTPUT_DIR, f"seed{budget}_n_{i}.txt")):
        continue

    seed = args.seed + i
    random.seed(seed)
    rg = np.random.RandomState(seed)
    # Initialize seeds
    e_seeds = list(rg.choice(G.nodes(), extra_seeds, replace=False))
    print('Seeds:', e_seeds)

    logging.debug('Extra Seeds:' + str(e_seeds))

    with torch.no_grad():
        acmodel = DQNTrainer(input_dim=input_dim, state_dim=action_dim, action_dim=action_dim, replayBuff=None, lr=LR,
                             use_cuda=use_cuda, gamma=args.gamma,
                             eta=eta, gcn_num_layers=gcn_layers, num_pooling=num_pooling, assign_dim=assign_dim,
                             assign_hidden_dim=assign_hidden_dim)

        load_checkpoint(acmodel, model_prefix=model_prefix)
        acmodel.eval()

        env = NetworkEnv(fullGraph=G, seeds=e_seeds, opt_reward=0, nop_r=args.nop_reward,
                         times_mean=args.times_mean_env, bad_reward=args.bad_reward, clip_max=args.max_reward,
                         clip_min=args.min_reward, normalize=args.norm_reward, weight_model=args.weight_model)

        env.reset(seeds=e_seeds)
        node_list = list(env.active.union(env.possible_actions))

        res = []

        start_time = time.time()
        for stps in range(budget):
            print("=" * 50, "step: ", stps, "=" * 50)

            _ = env.state
            s_embs = get_embeds(env.sub)
            if args.const_features:
                s = [node_attrs[node_list], env.state]
            else:
                s = [make_env_attrs_1(env=env, embs=s_embs, n=len(G), input_dim=input_dim)[node_list], env.state]

            possible_actions = [node_list.index(x) for x in env.possible_actions]

            state_embed, _ = acmodel.get_node_embeddings(nodes_attr=s[0], adj=s[1], nodes=possible_actions)
            l = list(env.possible_actions)
            possible_actions_embed = [s_embs[x] for x in l]

            actual_action, q = get_action_curr1(s, s_embs, l)

            res.append(actual_action)

            _, r, d, _ = env.step(actual_action)

            node_list = list(env.active.union(env.possible_actions))

            print("selected: ", actual_action)
            logging.info('Action:' + str(actual_action))

            torch.cuda.empty_cache()

        print('Chosen:', res, '\n')

    end_time = time.time()

    current, peak = tracemalloc.get_traced_memory()
    logging.info(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    gc.collect()

    with open(os.path.join(OUTPUT_DIR, f"seed{budget}_n_{i}.txt"), 'w') as f:
        for s in res:
            f.write("%d\n" % s)

    with open(os.path.join(OUTPUT_DIR, f"time_budget{budget}_n_{i}.txt"), 'w') as f:
        f.write("%f" % (end_time - start_time))

    with open(os.path.join(OUTPUT_DIR, f"mem_budget{budget}_n_{i}.txt"), 'w') as f:
        f.write("%f" % (peak / 10 ** 6))
