import argparse
import os


def parse(dataset, weight_model, budget, eps, num=5):
    format_eps = "%.1f" % eps
    reward = 0.
    elasped_time = 0.
    memory_usage = 0.
    for n in range(num):
        stat_file_path = os.path.join("result", dataset, weight_model, "multi_iter", f"budget{budget}_iter_{n}_stat_IC_{format_eps}.txt")
        elasped_time += float(open(stat_file_path, "r").readline().strip().split(" ")[0])
        memory_usage += float(open(stat_file_path, "r").readline().strip().split(" ")[1])
        reward += float(open(os.path.join("result", dataset, weight_model, f"coverage_{budget}_n_iter_{n}.txt")).read())
    output_path = os.path.join(OUTPUT_DIR, f"budget{budget}.txt")
    reward /= num
    elasped_time /= num
    memory_usage /= num
    with open(output_path, "w") as f:
        print(f"budget {budget}: reward: {reward} time: {elasped_time} memory: {memory_usage} MB")
        f.write(f"{reward} {elasped_time} {memory_usage}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-m", "--weight_model", required=True)
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.5, help="default=0.5")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join("result", args.dataset, args.weight_model)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    budgets = [20, 50, 100, 150, 200]
    for k in budgets:
        parse(args.dataset, args.weight_model, k, args.eps, args.num)
