import os
import os.path as osp
from collections import defaultdict
import glob
import random
import time
import pickle

import numpy as np
import networkx as nx
from networkx.algorithms.centrality import eigenvector_centrality

import torch
from torch_geometric.data import DataLoader, Data
from scipy.special import softmax

import utils


class ProblemError(Exception):

    def __init__(self):
        self.message = "Input an invalid problem"
        super().__init__(self.message)


def spread(fname, seed_nodes):
    node_to_add = 999999999999999
    if len(seed_nodes) == 0:
        return 0
    subgraph = nx.read_edgelist(fname, nodetype=int, data=(('weight', float),))
    for node in seed_nodes:
        subgraph.add_edge(node_to_add, node, weight=1)
    activated_nodes = nx.descendants(subgraph, node_to_add)
    return len(activated_nodes)


def expected_spread(seed_nodes, graph="facebook", model="IC", chunksize=50,
                    spread_dic=None, pool=None):
    if pool is None:
        print("haven't given a pool for MP")

    if model == "IC":
        fnames = glob.glob(f"{graph}/subgraph/subgraph_*")
    else:
        print("entered an incorrect model!")
        return

    seed_nodes = [seed_nodes for _ in range(len(fnames))]

    if spread_dic is not None and tuple(sorted(seed_nodes[0])) in spread_dic:
        return spread_dic[tuple(sorted(seed_nodes[0]))], spread_dic

    spreads = pool.starmap(spread, zip(fnames, seed_nodes), chunksize)
    if spread_dic:
        spread_dic[tuple(sorted(seed_nodes[0]))] = np.mean(spreads)
        return np.mean(spreads), spread_dic
    else:
        return np.mean(spreads)


def close_pool(pool):
    pool.close()
    pool.join()


def get_edge_list(graph):
    edges = graph.edges()
    source = []
    root = []
    [(source.append(u), root.append(v)) for u, v in edges]
    return [source, root]


def make_graph_features_for_encoder(graph, graph_name):
    if type(graph) == nx.Graph:
        print("not a di graph")
        return

    elif type(graph) == nx.DiGraph:
        try:
            with open(f"{graph_name}/graph_features", mode="rb") as f:
                features = pickle.load(f)
        except FileNotFoundError:
            features = {}
            out_degrees = [len(graph.out_edges(node)) for node in range(graph.number_of_nodes())]
            out_degree_max = np.max(out_degrees)
            out_degree_min = np.min(out_degrees)
            out_e_weights = [sum([graph[node][neighbour]["weight"] for _, neighbour in graph.out_edges(node)]) for node in
                             range(graph.number_of_nodes())]
            out_max_e_weight = max(out_e_weights)
            out_min_e_weight = min(out_e_weights)

            ev_values = eigenvector_centrality(graph, max_iter=1000)

            for node in range(graph.number_of_nodes()):
                features[node] = [(out_degrees[node] - out_degree_min) / (out_degree_max - out_degree_min), (
                        out_e_weights[node] - out_min_e_weight) / (out_max_e_weight - out_min_e_weight), ev_values[node]]
            with open(f"{graph_name}/graph_features", mode="wb") as f:
                pickle.dump(features, f)
        return features


def make_subgraph(graph, nodes):
    assert type(graph) == nx.DiGraph
    subgraph = nx.DiGraph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v, w) for u, v, w in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True))]
    subgraph.add_weighted_edges_from(edges_to_add)
    return subgraph


def get_best_embeddings(encoder, filepath, device):
    with open(filepath, mode='rb') as f:
        graph_data = pickle.load(f)

    graphs_ = [g.to(device) for g in graph_data if g.spread_label == 1]
    loader = DataLoader(graphs_, batch_size=len(graphs_))
    with torch.no_grad():
        embd = encoder.forward(next(iter(loader)))
    best_embedding = embd.to(device)
    return best_embedding


def moving_average(x, w):
    means = [np.mean(x[i:max(0, i - w):-1]) if i != 0 else x[0] for i in range(len(x))]
    return means


def get_good_subgraphs(features, labels):
    good_embeddings = [feature.reshape((1, features.shape[1])) for feature, label in zip(features, labels) if int(label) == 1]
    return good_embeddings


def get_label(score):
    if score >= 0.99:
        label = 1
    elif score >= 0.94:
        label = 2
    elif score >= 0.89:
        label = 3
    else:
        label = 4
    return label


def get_fixed_size_subgraphs(graph, good_seeds, num_samples, counts, BUDGET, size, graph_name, best_score,
                             graph_features, logger=None):
    subgraphs_partition = defaultdict(list)
    probs = np.array([counts[seed] for seed in good_seeds])
    probs = softmax(probs)
    logger.info(f"get fixed size subgraphs start: {num_samples} samples for 4 labels respectively.")
    start = time.time()
    good_seeds_propotion = 1
    discount_factor = 0.9
    count = 0
    stop_flag = False
    MAX_NUM_SAMPLE_TO_STOP_FACTOR = 50
    stop_threshold = num_samples * MAX_NUM_SAMPLE_TO_STOP_FACTOR
    while not stop_flag:
        single_start_time = time.time()
        seeds = np.random.choice(list(good_seeds), size=int(BUDGET * good_seeds_propotion), replace=False, p=probs).tolist()
        num_good_seeds = len(seeds)
        seeds += random.sample([n for n in graph.nodes() if n not in seeds], size - num_good_seeds)

        subgraph = make_subgraph(graph, seeds)
        subgraph, transformation = relabel_graph(subgraph, True)
        seeds = call_imm(graph_name, subgraph, BUDGET)
        seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
        score_ = utils.calculate_influence_spread(osp.join(graph_name, "edges.txt"), seeds)
        label = get_label(score_ / best_score)
        # do not discount the proportion if higher-quality subragphs are not generated enough
        if all([len(subgraphs_partition[i]) >= num_samples for i in range(1, label + 1)]):
            good_seeds_propotion *= discount_factor
            
        single_end_time = time.time()
        logger.info(f"sample {count}, good seeds proportion: {num_good_seeds / size:.2f}\tratio: {score_ / best_score:.2f}\tlabel: {label}\truntime: {(single_end_time - single_start_time):.2f} sec")
        
        count += 1
        stop_flag = all([len(subgraphs_list) == num_samples for label, subgraphs_list in subgraphs_partition.items()]) and len(subgraphs_partition) == 4 or count > stop_threshold
        
        if len(subgraphs_partition[label]) == num_samples:
            continue

        g = Data(edge_index=torch.LongTensor(get_edge_list(subgraph)), num_nodes=subgraph.number_of_nodes())
        g.y = torch.LongTensor([label])
        features = [graph_features[transformation[node]] if node in transformation else graph_features[node] for node in range(subgraph.number_of_nodes())]
        g.x = torch.FloatTensor(features)
        g.num_seeds = np.sum([seed in good_seeds for seed in seeds])    # ?
        g.graph_name = graph_name
        g.spread_label = int(score_ / best_score >= 0.95)
        g.score = score_
        subgraphs_partition[label].append(g)
        
    end = time.time()
    if count > num_samples * 50:
        logger.info(f"CAN NOT GENERATE CORRECT QUALITY SUBGRAPH!!")
        logger.info(f"class size 1: {len(subgraphs_partition[0])}\t2:{len(subgraphs_partition[2])}\t3:{len(subgraphs_partition[3])}\t4:{len(subgraphs_partition[4])}")
    logger.info(f"get fixed size subgraphs took {((end - start) / 60):.3f} minutes\n")
    return subgraphs_partition


def call_imm(grapg_name, subgraph, k):
    graph_tmp_dir, graph_path = utils.write_tmp_file(subgraph, grapg_name)
    seed_random = random.randint(1, 100000000)
    output_prefix = osp.join(osp.dirname(graph_path), osp.basename(graph_path)).replace(".txt", "")
    epsilon = 0.5
    IMM_PROGRAM = osp.join(osp.dirname(__file__), "IMM", "imm_discrete") # create soft link first
    command = f"time {IMM_PROGRAM} -dataset {graph_path} -k {k} -model IC -epsilon {epsilon} -output {output_prefix} -seed_random {seed_random} -training_for_gain 1"
    print("command imm ", command)
    res = os.system(command)
    assert res == 0, "IMM program error occurred."
    seed_path = graph_path.replace(".txt", f"_seeds_IC_{epsilon}.txt")
    seeds = [int(i) for i in open(seed_path).read().split("\n") if i]
    return seeds


def relabel_edgelist(roots, dests, unique):
    N = len(unique)
    desired_labels = set([i for i in range(N)])
    already_labeled = set([int(node) for node in unique if node < N])
    desired_labels = desired_labels - already_labeled
    transformation = {}
    reverse_transformation = {}
    for node in unique:
        if node >= N:
            transformation[node] = desired_labels.pop()
            reverse_transformation[transformation[node]] = node

    new_roots = [transformation[r] if r in transformation else r for r in roots]
    new_dests = [transformation[r] if r in transformation else r for r in dests]
    edge_list = [new_roots, new_dests]
    return edge_list, transformation, reverse_transformation, N


def relabel_graph(graph: nx.Graph, return_reverse_transformation_dic=False, return_forward_transformation_dic=False):
    """
    forward transformation has keys being original nodes and values being new nodes
    reverse transformation has keys being new nodes and values being old nodes
    """
    node_label = {nodeId: index for nodeId, index in zip(graph.nodes(), range(graph.number_of_nodes()))}
    graph = nx.relabel_nodes(graph, node_label)
    if return_forward_transformation_dic and return_reverse_transformation_dic:
        return graph, node_label, {v: k for k, v in node_label.items()}
    elif return_forward_transformation_dic:
        return graph, node_label
    elif return_reverse_transformation_dic:
        return graph, {v: k for k, v in node_label.items()}
    return graph
