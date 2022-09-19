def calculate_coverage(G, S):
    covered_set = set()
    for u in S:
        for v in G.neighbors(u):
            covered_set.add(v)
    return len(covered_set) / len(G)
