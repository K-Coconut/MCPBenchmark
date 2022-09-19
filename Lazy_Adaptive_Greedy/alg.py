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


def lazy_adaptive_greedy(main_graph, k, n=10):
    num_nodes = main_graph.number_of_nodes()
    isCovered = [False for _ in range(num_nodes)]
    solution_set = []
    set_nodes = main_graph.degree().items()
    all_nodes_queue = []
    for nodeId, init_gain in set_nodes:
        heappush(all_nodes_queue, (-init_gain, 0, nodeId))

    queue = []
    while len(all_nodes_queue) != 0 and len(queue) < k * n:
        marginal_gain, timestamp, nodeId = heappop(all_nodes_queue)
        heappush(queue, (marginal_gain, timestamp, nodeId))
    td = - nlargest(1, all_nodes_queue)[0][0]

    for itr in range(k):
        marginal_gain, timestamp, nodeId = heappop(queue)
        while -timestamp != itr:
            marginal_gain = calculate_marginal_gains(main_graph.neighbors(nodeId), isCovered)
            if marginal_gain > td:
                heappush(queue, (-marginal_gain, -itr, nodeId))
            else:
                heappush(all_nodes_queue, (-marginal_gain, -itr, nodeId))
            if len(queue) == 0:
                queue = all_nodes_queue
            marginal_gain, timestamp, nodeId = heappop(queue)
        solution_set.append(nodeId)
        select_node(nodeId, main_graph.neighbors(nodeId), isCovered)
    return solution_set
