from heapq import heappush, heappop, nlargest


def calculate_marginal_gains(neighbors, isCovered):
    degree = 0
    for nbr in neighbors:
        if not isCovered[nbr]:
            degree += 1
    return degree


def select_node(node, neighbors, isCovered):
    isCovered[node] = True
    for nbr in neighbors:
        isCovered[nbr] = True


def lazy_greedy(main_graph, k):
    num_nodes = main_graph.number_of_nodes()
    isCovered = [False for _ in range(num_nodes)]
    solution_set = []
    set_nodes = main_graph.degree().items()
    heap = []
    for nodeId, init_gain in set_nodes:
        heappush(heap, (-init_gain, 0, nodeId))

    for itr in range(k):
        marginal_gain, timestamp, nodeId = heappop(heap)
        while -timestamp != itr:
            marginal_gain = calculate_marginal_gains(main_graph.neighbors(nodeId), isCovered)
            heappush(heap, (-marginal_gain, -itr, nodeId))
            marginal_gain, timestamp, nodeId = heappop(heap)
        solution_set.append(nodeId)
        select_node(nodeId, main_graph.neighbors(nodeId), isCovered)
    return solution_set
