import ctypes
import os
import sys
import time
from tqdm import tqdm

sys.path.append( '%s/mvc_lib' % os.path.dirname(os.path.realpath(__file__)) )
from mvc_lib import MvcLib

sys.path.append( '%s/../datatracker' % os.path.dirname(os.path.realpath(__file__)) )
from graph_util import *

def gen_new_graphs(opt):
    print('generating new training graphs')
    sys.stdout.flush()
    api.ClearTrainGraphs()
    for i in tqdm(range(100)):
        while True:
            g = get_mvc_graph(g_undirected,prob_quotient=float(opt['prob_q']))
            assert len(g)
            if len(g) > 300:
                continue
            break
        api.InsertGraph(g, is_test=False)

def greedy(G):
    covered_set = set()
    numCoveredEdges = 0
    idxes = range(nx.number_of_nodes(G))
    idxes = sorted(idxes, key=lambda x: len(nx.neighbors(G, x)), reverse=True)
    pos = 0
    while numCoveredEdges < nx.number_of_edges(G):
        new_action = idxes[pos]
        covered_set.add(new_action)
        for neigh in nx.neighbors(G, new_action):
            if neigh not in covered_set:
                numCoveredEdges += 1
        pos += 1
    print('done')
    return len(covered_set)

if __name__ == '__main__':
    print(" ".join(sys.argv[1:]))
    api = MvcLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    g_undirected, _ = build_full_graph('%s/edges.txt' % opt['data_root'],'undirected')
    print(nx.number_of_nodes(g_undirected))
    print(nx.number_of_edges(g_undirected))
    # print(greedy(g_undirected))

    api.InsertGraph(g_undirected, is_test=True)

    # startup
    start_time = time.time()
    gen_graph_time = 0
    gen_new_graphs(opt)
    gen_graph_time += time.time() - start_time

    for i in range(10):
        api.lib.PlayGame(100, ctypes.c_double(1.0))
    api.TakeSnapshot()

    print('\n')
    eps_start = 1.0
    eps_end = 0.05
    eps_step = 10000.0
    best_r = nx.number_of_edges(g_undirected)
    best_t = 0
    for iter in tqdm(range(int(opt['max_iter']))):
        if iter and iter % 5000 == 0:
            tmp = time.time()
            gen_new_graphs(opt)
            gen_graph_time += time.time() - tmp
            print('\n')
        eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
        if iter % 10 == 0:
            api.lib.PlayGame(10, ctypes.c_double(eps))

        if iter % 300 == 0:
            frac = api.lib.Test(0)
            print('iter', iter, 'eps', eps, 'average pct of vc: ', frac)
            if frac < best_r:
                best_r = frac
                best_t = iter
                log_best_file = open('%s/best_model.txt' % opt['save_dir'], 'w')
                log_best_file.write("vc: %d\niter: %d" %(best_r, best_t))
                log_best_file.close()
            sys.stdout.flush()
            model_path = '%s/iter_%d.model' % (opt['save_dir'], iter)
            api.SaveModel(model_path)
            with open('%s/time_iter_%d.txt' % (opt['save_dir'], iter), 'w') as f:
                # total time, training time, generating graphs time
                print('total time: %f, training time: %f, generating graphs time: %f' %(time.time() - start_time, time.time() - start_time - gen_graph_time, gen_graph_time))
                f.write('%f %f %f' %(time.time() - start_time, time.time() - start_time - gen_graph_time, gen_graph_time))
        if iter % 1000 == 0:
            api.TakeSnapshot()

        api.lib.Fit()
