from collections import defaultdict
import numpy as np
import networkx as nx
import random
import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
import time
from networkx.algorithms.centrality import eigenvector_centrality


class ProblemError(Exception):

    def __init__(self):
        self.message = "Input an invalid problem"
        super().__init__(self.message)


def cover(graph: nx.Graph, selected_nodes):
    graph = graph.copy()
    quality = len(selected_nodes)
    for node in selected_nodes:
        nodes_covered = list(graph.neighbors(node))
        quality += len(nodes_covered)
        for nd in nodes_covered:
            if nd not in selected_nodes:
                graph.remove_node(nd)
    return quality

def calculate_cover(G, S):
    covered_set = set()
    for u in S:
        for v in G.neighbors(u):
            covered_set.add(v)
    return len(covered_set)

def calculate_degree(neighbors, is_covered):
    degree = 0
    for nbr in neighbors:
        if is_covered[int(nbr)] == False:
            degree += 1
    return degree


def select_node(node, neighbors, is_covered):
    is_covered[int(node)] = True
    for nbr in neighbors:
        is_covered[int(nbr)] = True


def greedy_mvc(main_graph, budget, set_nodes=None):
    if not set_nodes:
        set_nodes = sorted(main_graph.nodes())
    num_nodes = main_graph.number_of_nodes()

    is_covered = [False for _ in range(0, num_nodes)]

    solution_set = []

    while len(solution_set) != budget:
        denom = 0.0

        gains = [0 for _ in range(0, num_nodes)]
        for nd in set_nodes:
            gain = calculate_degree(main_graph.neighbors(nd), is_covered)
            gains[int(nd)] = gain
            denom += gain
        if np.sum(gains) == 0:
            break
        selection = -1
        max_gain = -1
        for nd in set_nodes:
            if gains[int(nd)] >= max_gain and nd not in solution_set:
                selection = nd
                max_gain = gains[int(nd)]
        solution_set.append(selection)
        select_node(selection, main_graph.neighbors(selection), is_covered)
    return solution_set


def prob_greedy_mvc(main_graph, set_nodes, delta):
    num_nodes = main_graph.number_of_nodes()
    set_nodes = sorted(set_nodes)
    is_covered = [False for _ in range(0, num_nodes)]

    solution_set = []

    while True:

        gains = [0 for _ in range(0, num_nodes)]
        for nd in set_nodes:
            gain = calculate_degree(main_graph.neighbors(nd), is_covered)
            gains[int(nd)] = gain
        if np.sum(gains) == 0:
            break
        selection = int(np.random.choice(set_nodes, 1, p=gains / np.sum(gains)))
        if selection in solution_set:
            print("node already in solution set")
            continue
        solution_set.append(selection)
        if gains[selection] / main_graph.number_of_edges() < delta:
            break
        select_node(selection, main_graph.neighbors(selection), is_covered)
    return solution_set


def calculate_marginal_gains(neighbors, isCovered):
    degree = 0
    for nbr in neighbors:
        if not isCovered[nbr]:
            degree += 1
    return degree

def lazy_adaptive_greedy(main_graph, k, n=10):
    from heapq import heappush, heappop, nlargest
    num_nodes = main_graph.number_of_nodes()
    isCovered = [False for _ in range(num_nodes)]
    solution_set = []
    set_nodes = main_graph.degree().items()
    H = []
    for nodeId, init_gain in set_nodes:
        heappush(H, (-init_gain, 0, nodeId))

    Q = []
    while len(H) != 0 and len(Q) < k * n:
        marginal_gain, timestamp, nodeId = heappop(H)
        heappush(Q, (marginal_gain, timestamp, nodeId))
    td = - nlargest(1, H)[0][0]

    for itr in range(k):
        if len(Q) == 0:
            Q = H
        marginal_gain, timestamp, nodeId = heappop(Q)
        while -timestamp != itr:
            marginal_gain = calculate_marginal_gains(main_graph.neighbors(nodeId), isCovered)
            if marginal_gain > td:
                heappush(Q, (-marginal_gain, -itr, nodeId))
            else:
                heappush(H, (-marginal_gain, -itr, nodeId))
            if len(Q) == 0:
                Q = H
            marginal_gain, timestamp, nodeId = heappop(Q)
        solution_set.append(nodeId)
        select_node(nodeId, main_graph.neighbors(nodeId), isCovered)
    return solution_set

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
    try:
        with open(f"{graph_name}/graph_features", mode="rb") as f:
            features = pickle.load(f)
    except FileNotFoundError:
        features = {}
        out_degrees = [len(list(graph.neighbors(node))) for node in range(graph.number_of_nodes())]
        out_degree_max = np.max(out_degrees)
        out_degree_min = np.min(out_degrees)

        ev_values = eigenvector_centrality(graph, max_iter=1000)

        for node in range(graph.number_of_nodes()):
            features[node] = [(out_degrees[node] - out_degree_min) / (out_degree_max - out_degree_min), ev_values[node]]
        with open(f"{graph_name}/graph_features", mode="wb") as f:
            pickle.dump(features, f)
    return features


def make_subgraph(graph, nodes):
    assert type(graph) == nx.Graph
    subgraph = nx.Graph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
    subgraph.add_edges_from(edges_to_add)
    return subgraph


def get_test_accuracy(predictions, targets):
    predictions = torch.argmax(predictions, dim=1)
    score = torch.sum(predictions == targets)
    return float(score / predictions.shape[0] * 100)


def get_best_embeddings(encoder, filepath, device):
    with open(filepath, mode='rb') as f:
        graph_data = pickle.load(f)
    encoder = encoder.to(device)

    graphs_ = [g.to(device) for g in graph_data if g.spread_label == 1]
    loader = DataLoader(graphs_, batch_size=len(graphs_))
    with torch.no_grad():
        embd = encoder.forward(next(iter(loader)))
    best_embedding = embd.cpu()
    return best_embedding


def moving_average(x, w):
    means = [np.mean(x[i:max(0, i - w):-1]) if i != 0 else x[0] for i in range(len(x))]
    return means


def get_good_subgraphs(features, labels):
    good_embeddings = [feature.reshape((1, features.shape[1])) for feature, label in zip(features, labels) if int(label) == 1]
    return good_embeddings


def get_label(score):
    if score >= 0.95:
        label = 1
    elif score >= 0.8:
        label = 2
    elif score >= 0.6:
        label = 3
    else:
        # original: only uses three classes
        label = 4
    return label


def get_fixed_size_subgraphs(graph, good_seeds, num_samples, BUDGET, size, best_score, graph_features, logger=None):
    subgraphs_partition = defaultdict(list)
    make_undirected = ToUndirected()
    
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
        seeds = np.random.choice(list(good_seeds), size=int(BUDGET * good_seeds_propotion), replace=False).tolist()
        num_good_seeds = len(seeds)
        seeds += random.sample([n for n in graph.nodes() if n not in seeds], size - num_good_seeds)

        subgraph = make_subgraph(graph, seeds)
        subgraph_, transformation = relabel_graph(subgraph, True)
        try:
            seeds = lazy_adaptive_greedy(subgraph_, BUDGET)
        except:
            seeds = greedy_mvc(subgraph_, BUDGET)
        seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
        score_ = calculate_cover(graph, seeds)
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
        
        g = Data(edge_index=torch.LongTensor(get_edge_list(subgraph_)), num_nodes=subgraph_.number_of_nodes())
        make_undirected(g)
        g.y = torch.LongTensor([label])
        features = [graph_features[transformation[node]] if node in transformation else graph_features[node] for node in range(subgraph_.number_of_nodes())]
        g.x = torch.FloatTensor(features)
        g.num_seeds = np.sum([seed in good_seeds for seed in seeds])
        g.spread_label = int(score_ / best_score >= 0.95)
        g.score = score_
        subgraphs_partition[label].append(g)
        end = time.time()
    
    end = time.time()
    if count > num_samples * 50:
        logger.info(f"CAN NOT GENERATE CORRECT QUALITY SUBGRAPH!!")
        logger.info(f"class size 1: {len(subgraphs_partition[1])}\t2:{len(subgraphs_partition[2])}\t3:{len(subgraphs_partition[3])}\t4:{len(subgraphs_partition[4])}")
    logger.info(f"get fixed size subgraphs took {((end - start) / 60):.3f} minutes\n")
    return subgraphs_partition


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