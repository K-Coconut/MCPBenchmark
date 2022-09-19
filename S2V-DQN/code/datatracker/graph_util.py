import networkx as nx
from tqdm import tqdm
import random


def build_full_graph(pathtofile, graphtype):
    node_dict = {}

    if graphtype == 'undirected':
        g = nx.Graph()
    elif graphtype == 'directed':
        g = nx.DiGraph()
    else:
        print('Unrecognized graph type .. aborting!')
        return -1

    f = open(pathtofile)
    sep = ' '
    for line in tqdm(f, unit='line'):
        if line.startswith('#'): continue
        entrices = line.strip().split(sep)
        if len(entrices) == 1 and sep not in line:
            sep = '\t'
            entrices = line.strip().split(sep)
        src_str = int(entrices[0])
        dst_str = int(entrices[1])
        if len(entrices) < 3:
            w = random.choice([0.1, 0.05])
        else:
            w = float(entrices[2])

        if src_str not in node_dict:
            node_dict[src_str] = len(node_dict)
            g.add_node(node_dict[src_str])
        if dst_str not in node_dict:
            node_dict[dst_str] = len(node_dict)
            g.add_node(node_dict[dst_str])

        src_idx = node_dict[src_str]
        dst_idx = node_dict[dst_str]

        c = 1
        if g.has_edge(src_idx, dst_idx):
            w += g[src_idx][dst_idx]['weight']
            c += g[src_idx][dst_idx]['count']

        g.add_edge(src_idx, dst_idx, weight=w, count=c)

    for edge in g.edges_iter(data=True):
        src_idx = edge[0]
        dst_idx = edge[1]
        w = edge[2]['weight']
        c = edge[2]['count']
        g[src_idx][dst_idx]['weight'] = w / c

    return g, node_dict


# fixme: prob-quotient value, w / pro_quotient too small
def get_mvc_graph(ig, prob_quotient=1):
    g = ig.copy()
    # flip coin for each edge, remove it if coin has value > edge probability ('weight')
    to_remove_edges = []
    for edge in g.edges_iter(data=True):
        src_idx = edge[0]
        dst_idx = edge[1]
        w = edge[2]['weight']
        coin = random.random()
        if coin > w / prob_quotient:
            to_remove_edges.append((src_idx, dst_idx))
    for edge in to_remove_edges:
        g.remove_edge(edge[0], edge[1])

    # get set of nodes in largest component
    cc = sorted(nx.connected_components(g), key=len, reverse=True)
    lcc = cc[0]

    # remove all nodes not in largest component
    numrealnodes = 0
    node_map = {}
    for node in g.nodes():
        if node not in lcc:
            g.remove_node(node)
            continue
        node_map[node] = numrealnodes
        numrealnodes += 1

    # re-create the largest component with nodes indexed from 0 sequentially
    g2 = nx.Graph()
    for edge in g.edges_iter(data=True):
        src_idx = node_map[edge[0]]
        dst_idx = node_map[edge[1]]
        w = edge[2]['weight']
        g2.add_edge(src_idx, dst_idx, weight=w)

    return g2


def visualize(g, pdfname='graph.pdf'):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(g, iterations=100)  # positions for all nodes
    nx.draw_networkx_nodes(g, pos, node_size=1)
    nx.draw_networkx_edges(g, pos)
    plt.axis('off')
    plt.savefig(pdfname, bbox_inches="tight")
