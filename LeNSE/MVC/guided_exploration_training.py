import os
import argparse
import pickle
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import torch

from functions import get_best_embeddings, moving_average
from rl_algs import GuidedDQN, DQN
from environment import GuidedExplorationEnv
import utils

def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', default="BrightKite")
    args.add_argument("-k", '--budget', default=100)
    args.add_argument("--gpu", type=int, default=0)
    args.add_argument("--num_eps", type=int, default=20)
    args.add_argument("--chunksize", type=int, default=28)
    args.add_argument("--subgraph_size", type=int, default=750)
    args.add_argument("--soln_budget", type=int, default=100)
    args.add_argument("--action_limit", type=int, default=1000)
    args.add_argument("--max_memory", type=int, default=20000)
    args.add_argument("--gnn_input", type=int, default=128)
    args.add_argument("--embedding_size", type=int, default=64)
    args.add_argument("--decay_rate", type=float, default=0.999975)
    args.add_argument("--ff_size", type=int, default=128)
    args.add_argument("--alpha", type=float, default=0.1)
    args.add_argument("--beta", type=float, default=0.1)
    args.add_argument("--input_size", type=int, default=10)
    args.add_argument("-b", "--batch_size", type=int, default=32)
    args.add_argument('--point_proportion', default=20, type=int, help="point proportion that training dataset takes up")
    opt = args.parse_args()
    
    logger = utils.get_logger("guided_exploration_training", os.path.join("log", f"guided_train_{opt.dataset}.log"))
    logger.info("=" * 150)
    logger.info(opt)
    utils.set_seed(seed=1)

    graph_name = os.path.join(opt.dataset, "train")
    encoder_name = "encoder"
    num_eps = opt.num_eps
    chunksize = opt.chunksize
    soln_budget = opt.soln_budget
    subgraph_size = opt.subgraph_size
    action_limit = opt.action_limit
    max_memory = opt.max_memory
    decay_rate = opt.decay_rate
    ff_size = opt.ff_size
    beta = opt.beta
    alpha = opt.alpha
    device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

    base_dir = os.path.join("data", graph_name)
    encoder = torch.load(os.path.join(base_dir, f"budget_{soln_budget}", "encoder", encoder_name + ".pth"), map_location=torch.device(device))
    encoder.to(device)
    node_feat_dim = encoder.gcn1.out_channels   # action_dim
    
    graph = nx.read_edgelist(os.path.join(base_dir, "edges.txt"), nodetype=int, create_using=nx.Graph())
    best_embeddings = get_best_embeddings(encoder, f"{base_dir}/budget_{soln_budget}/graph_data", device)
    embedding_size = best_embeddings.shape[1]
    dqn = GuidedDQN(gnn_input=node_feat_dim, state_dim=embedding_size,batch_size=opt.batch_size, decay_rate=decay_rate, ff_hidden=ff_size, gamma=0.95, max_memory=max_memory, device=device,
                    alpha=alpha)
    env = GuidedExplorationEnv(graph, soln_budget, subgraph_size, encoder, best_embeddings, graph_name, action_limit=action_limit, beta=beta, device=device)
    best_embedding = env.best_embedding_cpu.numpy()

    distances = []
    ratios = []
    rewards = []
    for episode in range(num_eps):
        logger.info(f"starting episode {episode + 1}")
        state = env.reset()
        done = False
        count = 0
        while not done:
            count += 1
            if count % 500 == 0:
                logger.info(f"actions count: {count}")
            action, state_for_buffer = dqn.act(state)
            next_state, reward, done = env.step(action)
            dqn.remember(state_for_buffer, reward, next_state[0], done)
            if count % 5 == 0:
                dqn.experience_replay()
            state = next_state

        if dqn.epsilon > dqn.epsilon_min:
            logger.info(f"Exploration rate currently at {dqn.epsilon:.3f}")
        final_dist = distance(env.subgraph_embedding, best_embedding)
        distances.append(-final_dist)
        logger.info(f"final distance of {final_dist}")
        logger.info(f"Ratio of {env.ratios[-1]:.3f}, sum of rewards {sum(env.episode_rewards)}\n")
        ratios.append(env.ratios[-1])
        plt.plot(env.episode_rewards)
        plt.savefig("ep_rewards.pdf")
        plt.clf()

        if (episode + 1) % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(env.ratios)
            ax1.plot(moving_average(env.ratios, 50))
            ax1.hlines(0.95, 0, len(env.ratios) - 1, colors="red")
            ax2.plot(distances)
            plt.savefig(f"{base_dir}/budget_{soln_budget}/{encoder_name}/dqn_training.pdf")
            plt.close(fig)

            with open(f"{base_dir}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="wb") as f:
                dqn_ = DQN(node_feat_dim, embedding_size, ff_size, 0.01, batch_size=0, device=device)
                dqn_.memory = ["hold"]
                dqn_.net = dqn.net
                pickle.dump(dqn_, f)
                del dqn_

            with open(f"{base_dir}/budget_{soln_budget}/{encoder_name}/guided_train_ratios", mode="wb") as f:
                pickle.dump(env.ratios, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(env.ratios)
    ax1.plot(moving_average(env.ratios, 50))
    ax1.hlines(0.95, 0, len(env.ratios) - 1, colors="red")
    ax2.plot(distances)
    plt.savefig(f"{base_dir}/budget_{soln_budget}/{encoder_name}/dqn_training.pdf")
    plt.close(fig)

    dqn.memory = ["hold"]
    dqn.batch_size = 0
    dqn_ = DQN(node_feat_dim, embedding_size, ff_size, 0.01, batch_size=0, device=device)
    dqn_.memory = dqn.memory
    dqn_.net = dqn.net
    with open(f"{base_dir}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="wb") as f:
        pickle.dump(dqn_, f)

    with open(f"{base_dir}/budget_{soln_budget}/{encoder_name}/guided_train_ratios", mode="wb") as f:
        pickle.dump(env.ratios, f)
