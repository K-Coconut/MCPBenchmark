import os
import sys
from pathlib import Path
import time

sys.path.append('%s/mvc_lib' % os.path.dirname(os.path.realpath(__file__)))
from mvc_lib import MvcLib

sys.path.append('%s/../datatracker' % os.path.dirname(os.path.realpath(__file__)))
from graph_util import *


def find_model_file(opt):
    max_n = int(opt['max_n'])
    min_n = int(opt['min_n'])
    log_file = '%s/log-%d-%d.txt' % (opt['save_dir'], min_n, max_n)

    best_r = 1000000
    best_it = -1
    with open(log_file, 'r') as f:
        for line in f:
            if 'average' in line:
                line = line.split(' ')
                it = int(line[1].strip())
                r = float(line[-1].strip())
                if r < best_r:
                    best_r = r
                    best_it = it
    assert best_it >= 0
    print('using iter=', best_it, 'with r=', best_r)
    return '%s/iter_%d.model' % (opt['save_dir'], best_it)


if __name__ == '__main__':
    print(" ".join(sys.argv[1:]))
    api = MvcLib(sys.argv)

    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    print("Loading dataset: ", opt['data_root'].split("/")[-1])
    g, _ = build_full_graph('%s/edges.txt' % opt['data_root'], 'undirected')
    print("n: %d\nm: %d" % (g.number_of_nodes(), g.number_of_edges()))
    api.InsertGraph(g, is_test=True)

    output_dir = opt['output_root']
    os.system("mkdir -p %s" % output_dir)
    best_model_iter = int(open("%s/best_model.txt" % opt['save_dir']).read().split("\n")[1].split(": ")[1])
    best_model_file = "%s/iter_%d.model" % (opt['save_dir'], best_model_iter)
    model = api.LoadModel(best_model_file)
    print("Best Model: iter_%d.model Loaded" % best_model_iter)

    budgets_to_test = [20, 50, 100, 150, 200]
    process_bar = tqdm(budgets_to_test)
    for budget in budgets_to_test:
        result_file = '%s/budget%d.txt' % (output_dir, budget)
        if Path(result_file).exists(): continue
        t1 = time.time()
        val, sol = api.GetBudgetedSol(0, nx.number_of_nodes(g), budget=budget)
        t2 = time.time()
        coverage = val / nx.number_of_nodes(g)
        f_out = open(result_file, 'w')
        f_out.write('%.8f %.6f\n' % (coverage, t2 - t1))
        f_out.write(str(sol[1:sol[0]]))
        f_out.close()
        process_bar.set_postfix(budget=budget, coverage=coverage)
        process_bar.update()
