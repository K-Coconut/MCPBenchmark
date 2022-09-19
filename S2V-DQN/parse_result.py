import os
import argparse

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dataset", required=True)
    args = args.parse_args()

    dataset = args.dataset
    DATA_ROOT = os.path.join("code", "realworld_s2v_mvc", "results", "testphase", dataset)
    OUTPUT_ROOT = os.path.join("..", "BenchmarksResult", "MCP", "S2V-DQN")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    budget = [20, 50, 100, 150, 200]
    for k in budget:
        file = os.path.join(DATA_ROOT, f"budget{k}.txt")
        coverage, runtime = open(file, 'r').readline().split(" ")
        with open(os.path.join(OUTPUT_ROOT, f"coverage{k}.txt"), 'w') as f:
            f.write(str(coverage))
        with open(os.path.join(OUTPUT_ROOT, f"time{k}.txt"), 'w') as f:
            f.write(str(runtime))
