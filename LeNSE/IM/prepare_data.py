import pickle
import argparse
import utils
import os
import time

from torch_geometric.data import DataLoader
import torch

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("-m", '--weight_model', default='TV', help='weight model')
args.add_argument("-k", '--budget', default=100)
args.add_argument("--gpu", type=int, default=2)
args.add_argument("-b", "--batch_size", type=int, default=128)
args.add_argument('--point_proportion', default=20, type=int, help="point proportion that training dataset takes up")
opt = args.parse_args()

dataset = opt.dataset
weight_model = opt.weight_model
encoder_graph_name = "lastfm"
budget = opt.budget
batch_size = opt.batch_size

logger = utils.get_logger("prepare_data", os.path.join("log", f"prepare_data_{dataset}_{weight_model}.log"))
logger.info(opt)

load_name = "graph_data"
save_name = "train_data"

train_size_proportion = opt.point_proportion
if train_size_proportion != 20:
    base_dir = os.path.join(
        "data", dataset, weight_model, "train_size_exp", f"train_size_{train_size_proportion}", "train", f"budget_{budget}"
    )
else:
    base_dir = os.path.join(
        "data", dataset, weight_model, "train", f"budget_{budget}"
    )

encoder = torch.load(os.path.join(base_dir, "encoder", "encoder.pth"))
device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"
encoder = encoder.to(device).eval()

with open(os.path.join(base_dir, load_name), mode="rb") as f:
    data = pickle.load(f)

data = [d.to(device) for d in data]
for d in data:
    d.y = d.y - 1
loader = DataLoader(data, batch_size=128)
logger.info(f"data size: {len(data)}")

start_time = time.time()
with torch.no_grad():
    features = []
    labels = []
    for batch in loader:
        embeddings = encoder(batch)
        features += embeddings.tolist()
        labels += batch.y.tolist()
        # labels += batch.spread_label
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    logger.info(f"labels: {labels.unique(return_counts=True)}")
    end_time = time.time()
    logger.info(f"runtime: {end_time - start_time:.3f}")
    with open(os.path.join(base_dir, "encoder", save_name), mode="wb") as f:
        pickle.dump((features.to("cpu"), labels.to("cpu")), f)
