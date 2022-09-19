from collections import defaultdict

from tqdm import tqdm


class Graph(object):
    def __init__(self):
        self.n = 0
        self.m = 0
        self.nodes = defaultdict(set)
        self.nodes_degree = {}

    def __len__(self):
        return self.n

    def size(self):
        return self.m

    def number_of_nodes(self):
        return self.n

    def degree(self):
        return self.nodes_degree

    def _init_degree(self):
        for n, neighbors in self.nodes.items():
            self.nodes_degree[n] = len(neighbors)

    def neighbors(self, nodeId):
        return self.nodes[nodeId]

    def read_edges(self, path):
        f = open(path, 'r')
        pb = tqdm(f, total=self.m, unit='line', desc='Loading Graph')
        m = 0
        id_map = {}
        sep = ' '
        for line in pb:
            if line.startswith('#'):
                continue
            line = line.strip().split(sep)
            if len(line) < 2:
                if line and '\t' in line[0]:
                    sep = '\t'
                    line = line[0].split(sep)
                else:
                    break
            m += 1
            source = int(line[0])
            target = int(line[1])
            if source not in id_map:
                id_map[source] = len(id_map)
            if target not in id_map:
                id_map[target] = len(id_map)

            self.nodes[id_map[source]].add(id_map[target])
            self.nodes[id_map[target]].add(id_map[source])

        self.n = len(self.nodes)
        self.m = m

        self._init_degree()
