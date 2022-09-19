def calculate_degree(neighbors, isCovered):
    degree = 0
    for nbr in neighbors:
        if not isCovered[nbr]:
            degree += 1
    return degree


def select_node(node, neighbors, isCovered):
    isCovered[node] = True
    for nbr in neighbors:
        isCovered[nbr] = True


def greedy(main_graph, k):
    num_nodes = main_graph.number_of_nodes()
    isCovered = [False for _ in range(0, num_nodes)]
    solution_set = []

    for itr in range(0, k):
        denom = 0.0

        gains = [0 for _ in range(0, num_nodes)]
        for nd in range(num_nodes):
            gain = calculate_degree(main_graph.neighbors(nd), isCovered)
            gains[nd] = gain
            denom += gain

        selection = -1
        max_gain = -1
        for nd in range(num_nodes):
            if gains[nd] >= max_gain:
                selection = nd
                max_gain = gains[nd]
        solution_set.append(selection)
        select_node(selection, main_graph.neighbors(selection), isCovered)
    return solution_set
