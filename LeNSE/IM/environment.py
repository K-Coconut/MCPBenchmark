import networkx as nx
from functions import relabel_graph, make_graph_features_for_encoder, relabel_edgelist, call_imm
import random
import torch
from torch_geometric.data import Data
import copy
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import time
import utils


class BaseEnvironment:

    def __init__(self, graph, solution_budget, subgraph_size, encoder, graph_name, Type="main", model="IC", chunksize=80, action_limit=5000, device="cpu", training=True):
        self.training = training
        self.device = device
        self.graph = graph
        self.solution_budget = solution_budget
        self.subgraph_size = subgraph_size
        self.encoder = encoder.to(self.device)
        self.graph_name = graph_name
        self.Type = Type
        self.model = model
        self.chunksize = chunksize
        self.rewards = []
        self.ratios = []
        self.all_nodes_set = set(self.graph.nodes())
        self.action_limit = action_limit
        self.neighbours = {}
        self.num_neighbours_sampled = 1

        self.edges = None
        self.roots = None
        self.unique = None
        self.dest = None
        self.node_feats = None
        self.optimal_score = None
        self.state = None
        self.episode_rewards = None
        self.subgraph_embedding = None
        self.graph_obj = None
        self.current_edge = None
        self.selected_nodes = None
        self.node_features_for_encoder = None
        self.selected_nodes = None
        self.num_action_taken = None
        self.fwd_transformation = None
        self.bwd_transformation = None

        self.get_optimal_scores()
        self.get_features_for_gnn()

        self.one_hop_neighbours = {}
        self.adjacent_nodes = {}
        self.get_one_hop_neighbours()

    def get_one_hop_neighbours(self):
        for node in self.graph.nodes():
            neighbours = [(u, v) for u, v in list(self.graph.out_edges(node))]
            neighbours += [(u, v) for u, v in list(self.graph.in_edges(node))]
            self.adjacent_nodes[node] = [u for u in list(self.graph.neighbors(node)) + list(self.graph.predecessors(node))]
            self.adjacent_nodes[node] = set(self.adjacent_nodes[node])
            self.one_hop_neighbours[node] = neighbours

    def get_optimal_scores(self):
        if not self.training:
            # no need to report the ratio in the exp
            return
        seed_file = os.path.join("data", self.graph_name, f"budget_{self.solution_budget}", "score_and_seeds")
        with open(seed_file, mode="rb") as f:
            good_seeds, best_score, counts = pickle.load(f)
            self.optimal_score = best_score

    def run_imm_on_final_subgraph(self):
        graph, transformation = relabel_graph(self.state, True)
        graph_name = os.path.join("data", self.graph_name)
        seeds = call_imm(graph_name, graph, self.solution_budget)
        seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
        return seeds

    def get_state_embedding(self):
        with torch.no_grad():
            self.graph_obj = self.get_graph_obj_for_encoder()
            subgraph_embedding, node_feats = self.encoder.forward(self.graph_obj, return_node_embedding=False, return_both=True)
            self.subgraph_embedding = subgraph_embedding.cpu().numpy()
            self.node_feats = node_feats.cpu()

    def get_new_state(self, action: tuple):
        node_to_remove, node_to_add = action
        assert node_to_remove in self.selected_nodes, "node to remove not in selected nodes"
        assert node_to_add not in self.selected_nodes, "node to add already in selected nodes"
        assert node_to_add in self.unique, "node not in one hop neighbourhood"

        selected_nodes = [n for n in self.selected_nodes if n != node_to_remove] + [node_to_add]
        self.selected_nodes = set(selected_nodes)
        assert len(self.selected_nodes) == self.subgraph_size, "selected node length does not equal subgraph size!!!"
        self.edges = []
        for node in self.selected_nodes:
            self.edges += self.one_hop_neighbours[node]
        self.roots = []
        self.dest = []
        [(self.roots.append(u), self.dest.append(v)) for u, v in self.edges]
        self.unique = set(self.roots + self.dest)
        self.get_neighbours()

    def get_graph_obj_for_encoder(self):
        edge_list, fwd_transformation, transformation, N = relabel_edgelist(self.roots, self.dest, self.unique)
        self.fwd_transformation = fwd_transformation
        self.bwd_transformation = transformation
        g = Data(edge_index=torch.LongTensor(edge_list), num_nodes=N)
        features = [self.node_features_for_encoder[transformation[node]] if node in transformation else self.node_features_for_encoder[node] for node in range(N)]
        g.x = torch.FloatTensor(features)
        g.batch = None
        g = g.to(self.device)
        return g

    def get_features_for_gnn(self):
        self.node_features_for_encoder = make_graph_features_for_encoder(self.graph, os.path.join("data", self.graph_name))

    def get_tensors(self):
        output, mapping, reverse_mapping = self.get_tensors_and_maps()
        return output, mapping, reverse_mapping

    def get_tensors_and_maps(self):
        count = 0
        mapping = {}
        reverse_mapping = {}
        features = []
        state = self.subgraph_embedding.squeeze().tolist()
        for node in self.neighbours:
            neighbours = self.neighbours[node]
            for neighbour in neighbours:
                if node in self.fwd_transformation:
                    node_id = int(self.fwd_transformation[node])
                else:
                    node_id = node
                if neighbour in self.fwd_transformation:
                    neighbour_id = int(self.fwd_transformation[neighbour])
                else:
                    neighbour_id = neighbour

                node_feat = self.node_feats[node_id]
                neighbour_feat = self.node_feats[neighbour_id]
                features += [state + node_feat.tolist() + neighbour_feat.tolist()]
                mapping[count] = (node, neighbour)
                reverse_mapping[(node, neighbour)] = count
                count += 1

        features = np.array(features)
        return features, mapping, reverse_mapping

    def get_initial_subgraph(self):
        self.selected_nodes = set(random.sample(self.graph.nodes(), self.subgraph_size))
        assert len(self.selected_nodes) == self.subgraph_size, "selected node length does not equal subgraph size!!!"
        self.edges = []
        for node in self.selected_nodes:
            self.edges += self.one_hop_neighbours[node]
        self.roots = []
        self.dest = []
        [(self.roots.append(u), self.dest.append(v)) for u, v in self.edges]
        self.unique = set(self.roots + self.dest)
        self.get_neighbours()

    def make_subgraph(self):
        edges_to_add = []
        self.state = nx.DiGraph()
        for node in self.selected_nodes:
            self.state.add_node(node)
            edges_to_add += [(u, v, self.graph[u][v]['weight']) for u, v in list(self.graph.out_edges(node)) + list(self.graph.in_edges(node))]
        self.state.add_weighted_edges_from(edges_to_add)

    def get_neighbours(self):
        self.neighbours = {}
        unselected_nodes = [node for node in self.unique if node not in self.selected_nodes]
        for node in self.selected_nodes:
            neighbours = [n for n in self.adjacent_nodes[node] if n not in self.selected_nodes]
            if len(neighbours) >= self.num_neighbours_sampled:
                neighbour = random.sample(neighbours, k=self.num_neighbours_sampled)

            else:
                neighbour = random.sample(unselected_nodes, self.num_neighbours_sampled)

            self.neighbours[node] = neighbour


class GraphModification(BaseEnvironment):
    def __init__(self, graph, solution_budget, subgraph_size, encoder, best_embedding, graph_name, Type="main", model="IC", chunksize=80, action_limit=5000, beta=1, device="cpu", training=True):

        super(GraphModification, self).__init__(graph, solution_budget, subgraph_size, encoder, graph_name, Type, model, chunksize, action_limit, device=device, training=training)

        self.best_embedding = None
        if best_embedding is not None:
            self.best_embedding = best_embedding
            self.best_embedding = torch.mean(self.best_embedding, dim=0).reshape((1, self.best_embedding.shape[1]))
            self.best_embedding_cpu = copy.deepcopy(self.best_embedding.cpu())
            self.best_embedding = self.best_embedding.to(self.device)

        self.beta = beta
        self.l_start = None
        self.l_end = None
        self.h_start = None
        self.h_end = None
        self.lense_times = []
        self.heuristic_times = []

    def reset(self):
        self.neighbours = {}
        self.episode_rewards = []
        self.num_action_taken = 0
        self.l_start = time.time()
        self.get_initial_subgraph()
        self.get_state_embedding()
        state = self.get_tensors()  # state = graph embedding dim + node feat dim, state = embedding_dim + node_dim * 2
        return state

    def step(self, action):
        self.num_action_taken += 1
        self.get_new_state(action)
        self.get_state_embedding()
        done = self.get_done()
        reward = self.get_reward()
        state = self.get_tensors()
        if done:
            self.make_subgraph()
            seeds = self.run_imm_on_final_subgraph()
            graph_path = os.path.join("data", self.graph_name, "edges.txt")
            score = utils.calculate_influence_spread(graph_path, seeds)
            ratio = score / self.optimal_score
            if ratio >= 0.95:
                reward += 1
            self.ratios.append(ratio)
            self.l_end = time.time()
            self.lense_times.append((self.l_end - self.l_start) / 60)
        if not done:
            self.episode_rewards.append(reward)
        return state, reward, done

    def get_reward(self):
        curr_emb = torch.FloatTensor(self.subgraph_embedding)
        distance = torch.norm((curr_emb - self.best_embedding_cpu), p=2, dim=1)
        if len(distance) > 1:
            distance = torch.min(distance)
        return self.beta * float(-distance)

    def get_done(self):
        return self.num_action_taken == self.action_limit


class TestEnv(GraphModification):
    def __init__(self, graph, solution_budget, subgraph_size, encoder, graph_name, Type="main", model="IC", chunksize=80, action_limit=1000, beta=1, device="cpu"):

        super(TestEnv, self).__init__(graph, solution_budget, subgraph_size, encoder, None, graph_name, Type, model, chunksize, action_limit, beta, device, training=False)

    def step(self, action):
        self.num_action_taken += 1
        self.get_new_state(action)
        self.get_state_embedding()
        done = self.get_done()
        state = self.get_tensors()
        if done:
            self.l_end = time.time()
            self.make_subgraph()
            self.lense_times.append((self.l_end - self.l_start) / 60)
        return state, done
    
    def run_imm_on_final_subgraph(self, budget, num_rr_set=1e5):
        graph, transformation = relabel_graph(self.state, True)
        graph_name = os.path.join("data", self.graph_name)
        st = time.time()
        seeds = call_imm(graph_name, graph, budget)
        runtime = time.time() - st
        seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
        graph_path = os.path.join("data", self.graph_name, "edges.txt")
        coverage = utils.calculate_influence_spread(graph_path, seeds, num_rr_set)
        return coverage, runtime, seeds
        


class TestEnvMultiBudget(TestEnv):

    def __init__(self, graph, solution_budget, subgraph_size, encoder, graph_name, action_limit=1000, beta=1, cuda=False, budgets=None):

        if budgets is None:
            self.budgets = [5, 10, 20, 50, 75]
        else:
            self.budgets = budgets

        if solution_budget not in self.budgets:
            self.budgets.append(solution_budget)

        self.optimal_scores = {}
        super(TestEnvMultiBudget, self).__init__(graph, solution_budget, subgraph_size, encoder, graph_name, "main", "IC", action_limit=action_limit, beta=beta, cuda=cuda)
        self.ratios = {}
        self.times = {}
        for budget in self.budgets:
            self.ratios[budget] = []
            self.times[budget] = []

    def get_optimal_scores(self):
        for budget in self.budgets:
            seed_file = os.path.join("data", self.graph_name, f"budget_{budget}", "score_and_seeds")
            with open(seed_file, mode="rb") as f:
                good_seeds, best_score, counts = pickle.load(f)
                self.optimal_scores[budget] = best_score

    def step(self, action):
        self.num_action_taken += 1
        self.get_new_state(action)
        self.get_state_embedding()
        done = self.get_done()
        reward = 0
        state = self.get_tensors()
        if done:
            self.make_subgraph()
            self.get_ratios()
        return state, reward, done

    def get_ratios(self):
        graph, transformation = relabel_graph(self.state, True)
        graph_path = os.path.join("data", self.graph_name, "edges.txt")
        for budget in self.budgets:
            start = time.time()
            seeds = call_imm(graph, budget)
            end = time.time()
            self.times[budget].append((end - start) / 60)
            seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
            score = utils.calculate_influence_spread(graph_path, seeds)
            self.ratios[budget].append(score / self.optimal_scores[budget])


class GuidedExplorationEnv(GraphModification):
    def __init__(self, graph, solution_budget, subgraph_size, encoder, best_embedding, graph_name, action_limit=5000, beta=1, device="cpu"):

        self.optimal_seeds = None
        super(GuidedExplorationEnv, self).__init__(graph, solution_budget, subgraph_size, encoder, best_embedding, graph_name, action_limit=action_limit, beta=beta, device=device)

    def get_optimal_scores(self):
        seed_file = os.path.join("data", self.graph_name, f"budget_{self.solution_budget}", "score_and_seeds")
        with open(seed_file, mode="rb") as f:
            good_seeds, best_score, counts = pickle.load(f)
            self.optimal_seeds = good_seeds
            self.optimal_score = best_score

    def get_tensors_and_maps(self):
        count = 0
        mapping = {}
        reverse_mapping = {}
        features = []
        optimal = []
        state = self.subgraph_embedding.squeeze().tolist()
        for node in self.neighbours:
            neighbours = self.neighbours[node]
            for neighbour in neighbours:
                if node in self.fwd_transformation:
                    node_id = int(self.fwd_transformation[node])
                else:
                    node_id = node
                if neighbour in self.fwd_transformation:
                    neighbour_id = int(self.fwd_transformation[neighbour])
                else:
                    neighbour_id = neighbour
                if node not in self.optimal_seeds and neighbour in self.optimal_seeds:
                    optimal.append(count)
                node_feat = self.node_feats[node_id]
                neighbour_feat = self.node_feats[neighbour_id]
                features += [state + node_feat.tolist() + neighbour_feat.tolist()]
                mapping[count] = (node, neighbour)
                reverse_mapping[(node, neighbour)] = count
                count += 1

        features = np.array(features)
        return features, mapping, reverse_mapping, optimal

    def get_tensors(self):
        output, mapping, reverse_mapping, optimal = self.get_tensors_and_maps()
        return output, mapping, reverse_mapping, optimal
