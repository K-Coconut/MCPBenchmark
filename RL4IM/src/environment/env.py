import numpy as np
import networkx as nx
import math
import random
import argparse
import pulp
import os
from src.IC import runIC_repeat
from src.agent.baseline import *

import time


class Environment(object):
    '''
    Environment for influence maximization
    G is a nx graph
    state is 1xN binary array,
    -- 1st row: invited in previous main step and came (=1), 
    -- 2nd row: invited but not come (=1); 
    -- 3rd row: invited in previous sub step (=1) or not (=0) --- only updated outside environment (in rl4im.py: greedy_action_GCN() and memory store step)
    '''
    def __init__(self, mode='train', T=20, budget=5, propagate_p = 0.1, l=0.05, d=1, q=1, cascade='IC', num_simul=1000, graphs=None, name='MVC', args=None):
        self.args = args
        self.name = name
        self.G = graphs[0] 
        self.graph_init = self.G  

        self.graphs = graphs
        self.mode = mode
        self.N = len(self.G.g)  
        self.budget = budget
        # self.A = nx.to_numpy_matrix(self.G.g)
        self.propagate_p = propagate_p
        self.l = l
        self.d = d
        self.q = q
        self.T = T
        self.cascade = cascade
        self.num_simul = self.args.num_simul_train
        self.t = 0
        self.done = False
        self.reward = 0
        self.state = np.zeros((1, self.N))
        self.observation = self.state
        nx.set_node_attributes(self.G.g, 0, 'attr')

    def step(self, i, pri_action, sec_action, reward_type=0):
        '''
        pri_action is a list, sec_action is an int
        reward type categories, example seed nodes before {1, 2, 3}, new node x
        0: reward0 = f({1, 2, 3, x}) - f({1, 2, 3})
        1: reward1 = f({x}) - f({ })
        2: reward2 = (reward0+reward1)/2
        3: use probabilty q 
        '''

        #compute reward as marginal contribution of a node
        if self.mode == 'train':
            # [seeds.append(v) for v in range(self.N) if (self.state[0][v]==1)] # always empty
            influece_without, influence_with = 0, 0
            while influece_without >= influence_with:
                influece_without = self.run_cascade(seeds=pri_action, cascade=self.cascade, sample=self.num_simul)
                influence_with = self.run_cascade(seeds=pri_action + [sec_action], cascade=self.cascade, sample=self.num_simul)
            self.reward = self.q*(influence_with - influece_without)
            self.reward = self.reward/self.N*100   ####


        #update next_state and done      
        # if i%self.budget == 0:
        #a primary step
        invited = pri_action
        present = invited
        for v in present:
            self.state[0][v]=1
        if i == self.T:
            next_state = None
            self.done = True
        else:
            next_state = self.state.copy()
            self.done = False
        # else:
        # #a secondary step
        #     next_state = self.state.copy()
        #     self.done = False

        if i == self.T:  
            next_state = None
            self.done = True

        return next_state, self.reward, self.done
            
    def run_cascade(self, seeds, cascade='IC', sample=1000):
        if cascade == 'IC':
            reward, _ = runIC_repeat(self.G.g, seeds, sample=sample)
        else:
            assert(False)
        return reward

    def f_multi(self, x):
        s=list(x) 
        val = self.run_cascade(seeds=s, cascade=self.cascade, sample=self.args.greedy_sample_size)
        return val
 
    #the simple state transition process
    def transition(self, invited):#q is probability being present
        return invited

    def reset(self, g_index=0, mode='train'):
        self.mode = mode
        if mode == 'test': 
            self.G = self.graphs[g_index]
        else:
            self.G = self.graphs[g_index]
        self.N = len(self.G.g)
        # self.A = nx.to_numpy_matrix(self.G.g)
        self.t = 0
        self.done = False
        self.reward = 0
        self.state = np.zeros((1, self.N))
        self.observation = self.state
        nx.set_node_attributes(self.G.g, 0, 'attr')

    def get_state(self, g_index):
        curr_g = self.graphs[g_index]
        available_action_mask = np.array([1] * curr_g.cur_n + [0] * (curr_g.max_node_num - curr_g.cur_n))

        # padding the state for storing
        obs_padding = self.observation.copy()
        if self.args.model_scheme == 'type1':
            padding = np.repeat(np.array([-1] * (curr_g.max_node_num - curr_g.cur_n))[None, ...], self.observation.shape[0], axis=0)
            obs_padding = np.concatenate((self.observation.copy(), padding), axis=-1)
        return self.observation.copy().squeeze(), obs_padding.squeeze(), available_action_mask

    def try_remove_feasible_action(self, feasible_actions, sec_action):
        try:
            feasible_actions.remove(sec_action)
            return feasible_actions
        except Exception:
            pass
        finally:
            return feasible_actions

    def prune_noisy_nodes(self, N=10):
        n_edge_weight = []
        G = self.G.g
        for u in G.nodes:
            w = 0.
            for v in G.neighbors(u):
                w += G.get_edge_data(u, v)['weight']
            w = G.out_degree[u] * w
            n_edge_weight.append(w)

        # 2-hop edge weight
        gamma = 0.5
        tmp_weight = n_edge_weight.copy()
        for u in G.nodes:
            w = n_edge_weight[u - 1]
            for v in G.neighbors(u):
                w += gamma * n_edge_weight[v - 1]
                for j in G.neighbors(v):
                    w += gamma * gamma * n_edge_weight[j - 1]
            tmp_weight[u - 1] = w
        n_edge_weight2 = tmp_weight

        kN = self.args.budget * N
        good_nodes_index = np.argpartition(n_edge_weight2, -kN)[-kN:]
        good_nodes = [i for i in good_nodes_index]  # node id starts from 1
        return good_nodes
