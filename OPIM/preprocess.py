import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-m", "--weight_model", required=True)
args = parser.parse_args()

g = os.path.join("..", "data", args.dataset, "IM", args.weight_model, "edges.txt")
if not os.path.exists(g):
    raise Exception(f"{g} does not exist.\nPlease execute preprocess script under data directory to generate {g}")
command = f"./OPIM1.1.o -func=0 -gname={args.dataset} -wmodel={args.weight_model} -mode=w"
print(command)
os.system(command)
