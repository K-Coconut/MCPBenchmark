import argparse
import os
import utils

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("--soln_budget", type=int, default=100)
    args.add_argument("-n", "--n_iter", type=int, default=1)
    args.add_argument("-f", "--action_factor", type=int, default=5)
    opt = args.parse_args()
    
    logger = utils.get_logger("parse_result", os.path.join("log", "parse_result.log"))
    budgets = [20, 50, 100, 150, 200]
    graph_name = opt.dataset
    encoder_name = "encoder"
    soln_budget = opt.soln_budget
    action_factor = opt.action_factor
    n_iter = opt.n_iter

    test_graph_name = os.path.join(opt.dataset, "test")
    output_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/"
    result_dir = f"results/{graph_name}/budget_{soln_budget}/"
    
    os.makedirs(result_dir, exist_ok=True)
    
    logger.info("=" * 20)
    logger.info(f"{opt.dataset}")

    for k in budgets:
        coverage, runtime, memory = 0, 0, 0
        for episode in range(n_iter):
            result_file = os.path.join(output_dir, f"budget{k}_factor_{action_factor}_iter_{episode}.txt")
            if not os.path.exists(result_file):
                print(f"{result_file} not exists")
                continue
            _coverage, _runtime, _memory = open(result_file).read().strip().split(" ")
            coverage += float(_coverage)
            runtime += float(_runtime)
            memory += float(_memory)
        coverage /= n_iter
        runtime /= n_iter
        memory /= (n_iter * 10 ** 6)
        logger.info(f"budget {k}: {coverage:.5f} {runtime:.3f} {memory:.3f}")
        with open(os.path.join(result_dir, f"budget{k}.txt"), "w") as f:
            f.write(f"{coverage:.5f} {runtime:.3f} {memory:.3f}")
    