import sys
import json

from networkx.algorithms import bipartite
from networkx.readwrite import json_graph


# changes the graph
def evaluate(graph, selected_nodes):
    quality = 0
    for node in selected_nodes:
        nodes_covered = graph.neighbors(node)
        quality += len(nodes_covered)
        for nd in nodes_covered:
            graph.remove_node(nd)
    return quality


# MCP coverage
def cal_bp_coverage(graph, selected_nodes):
    set_nodes, element_nodes = bipartite.sets(graph)
    nodes_in_element_part = []

    for node in selected_nodes:
        nodes_in_element_part.extend(graph.neighbors(node))

    nodes_in_element_part = set(nodes_in_element_part)
    return len(nodes_in_element_part) / len(element_nodes)