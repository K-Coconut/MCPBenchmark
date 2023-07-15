import argparse
import os
import utils

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", '--dataset', required=True)
    args.add_argument("-m", '--weight_model', default='TV', help='weight model')
    args.add_argument("--soln_budget", type=int, default=100)
    args.add_argument("-n", "--n_iter", type=int, default=1)
    args.add_argument("-f", "--action_factor", type=int, default=5)
    args.add_argument('--point_proportion', default=20, type=int, help="point proportion that training dataset takes up")
    args.add_argument('--multi_epoch', action='store_true', help="tested with trained ckpt of different epoch")
    opt = args.parse_args()
    
    logger = utils.get_logger("parse_result", os.path.join("log", "parse_result.log"))
    budgets = [20, 50, 100, 150, 200]
    graph_name = os.path.join(opt.dataset, opt.weight_model)
    encoder_name = "encoder"
    soln_budget = opt.soln_budget
    action_factor = opt.action_factor
    n_iter = opt.n_iter
    train_size_proportion = opt.point_proportion
    if train_size_proportion != 20:
        test_graph_name = os.path.join(opt.dataset, opt.weight_model, "train_size_exp", f"train_size_{train_size_proportion}", "test")
        output_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/"
        result_dir = f"results/{graph_name}/train_size_exp/train_size_{train_size_proportion}/budget_{soln_budget}/"
        budgets = [20]
    else:
        test_graph_name = os.path.join(opt.dataset, opt.weight_model, "test")
        output_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/"
        result_dir = f"results/{graph_name}/budget_{soln_budget}/"
    
    os.makedirs(result_dir, exist_ok=True)
    
    logger.info("=" * 20)
    logger.info(f"{opt.dataset} {opt.weight_model}")
    if not opt.multi_epoch:
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
    else:
        k = 20
        num_epochs = 300
        for epoch in range(1, num_epochs + 1):
            output_dir = f"data/{test_graph_name}/budget_{soln_budget}/{encoder_name}/results/train_epoch_{epoch}"
            result_dir = f"results/{graph_name}/train_time_exp/budget_{soln_budget}/"
            os.makedirs(result_dir, exist_ok=True)
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
            logger.info(f"trained epoch {epoch} budget {k}: {coverage:.5f} {runtime:.3f} {memory:.3f}")
            with open(os.path.join(result_dir, f"budget{k}_epoch_{epoch}.txt"), "w") as f:
                f.write(f"{coverage:.5f} {runtime:.3f} {memory:.3f}")