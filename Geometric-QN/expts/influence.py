import networkx as nx
import numpy as np
from icm import sample_live_icm, indicator, make_multilinear_objective_samples
from utils import greedy
from multiprocessing import Process, Manager
import random

PROP_PROBAB = 0.1
BUDGET = 10
PROCESSORS = 8
SAMPLES = 100


def multi_to_set(f, g):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''

    def f_set(S):
        return f(indicator(S, len(g)))

    return f_set


def influence(graph, full_graph, weight_model, samples=SAMPLES):
    for u, v in graph.edges():
        if weight_model == 'CONST':
            graph[u][v]['p'] = PROP_PROBAB
        elif weight_model == 'TV':
            graph[u][v]['p'] = random.choice([0.1, 0.01, 0.001])
        elif weight_model == 'WC':
            graph[u][v]['p'] = 1 / graph.in_degree(v)
        else:
            raise Exception('Wrong weight model')

    def genoptfunction(graph, samples=1000):
        live_graphs = sample_live_icm(graph, samples)
        f_multi = make_multilinear_objective_samples(live_graphs, list(graph.nodes()), list(graph.nodes()),
                                                     np.ones(len(graph)))
        f_set = multi_to_set(f_multi, graph)
        return f_set

    f_set = genoptfunction(graph, samples)
    S, obj = greedy(list(range(len(graph))), BUDGET, f_set)

    f_set1 = genoptfunction(full_graph, samples)
    opt_obj = f_set1(S)

    return opt_obj, obj, S


def parallel_influence(graph, full_graph, times, samples=SAMPLES, influence=influence, weight_model="CONST"):
    def influence_wrapper(l, g, fg, s, influence=influence, weight_model=weight_model):
        ans = influence(g, fg, weight_model, s)
        l.append(ans[0])

    l = Manager().list()
    processes = [Process(target=influence_wrapper, args=(l, graph, full_graph, samples)) for _ in range(times)]
    i = 0
    while i < len(processes):
        j = i + PROCESSORS if i + PROCESSORS < len(processes) else len(processes) - 1
        ps = processes[i:j]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        i += PROCESSORS
    l = list(l)
    return np.mean(l)
