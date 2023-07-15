import os
import argparse
import pickle
import random
import numpy as np
import networkx as nx

from torch_geometric.loader import DataLoader
import torch

from pytorch_metric_learning import losses, miners
from pytorch_tools import EarlyStopping
from networks import KPooling, GNN
import utils

args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("--ratio", default=0.8)
args.add_argument("--embedding_size", type=int, default=30)
args.add_argument("--temperature", default=0.1)
args.add_argument("--output_size", type=int, default=5)
args.add_argument("-b", "--batch_size", default=128, type=int)
args.add_argument("-l", "--learning_rate", default=0.001, type=float)
args.add_argument("--pooling", default=True, type=bool)
args.add_argument("--metric", default="distance")
args.add_argument('--budget', default=100)
args.add_argument('--gpu', default=0)
args.add_argument('--point_proportion', default=20, type=int, help="point proportion that training dataset takes up")
opt = args.parse_args()

utils.set_seed(1)

os.makedirs("log", exist_ok=True)
logger = utils.get_logger("embedding training", os.path.join("log", f"embedding_training_{opt.dataset}.log"))
logger.info("=" * 150)
logger.info(opt)

dataset = opt.dataset
pooling = opt.pooling
embedding_size = opt.embedding_size
temperature = opt.temperature
output_size = opt.output_size
budget = opt.budget
batch_size = opt.batch_size
learning_rate = opt.learning_rate
metric = opt.metric

device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"
train_ratio = 0.8

graph_dir = os.path.join("data", dataset, "train")
with open(f"{graph_dir}/budget_{budget}/graph_data", mode="rb") as f:
    data = pickle.load(f)
random.shuffle(data)
data = data[:1500]
data = [d.to(device) for d in data]
n = int(len(data) * train_ratio)
logger.info(f"Entire dataset size: {len(data)}\ttraining size ratio: {train_ratio}")

train_data = data[:n]
val_data = data[n:]
del data

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

if pooling:
    encoder = KPooling(train_ratio, 2, embedding_size, output_size).to(device)
else:
    encoder = GNN(2, embedding_size, output_size).to(device)
optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
loss_fn = losses.NTXentLoss(temperature)
miner = miners.MultiSimilarityMiner()
es = EarlyStopping(patience=10, percentage=False)

losses = []
val_losses = []
for epoch in range(1000):
    epoch_train_loss = []
    epoch_val_loss = []
    for count, batch in enumerate(train_loader):
        optimiser.zero_grad()
        inputs = encoder.forward(batch)
        hard_pairs = miner(inputs, batch.y)
        loss = loss_fn(inputs, batch.y, hard_pairs)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimiser.step()

    for batch in val_loader:
        with torch.no_grad():
            inputs = encoder.forward(batch)
            loss = loss_fn(inputs, batch.y)
            epoch_val_loss.append(loss.item())

    losses.append(np.mean(epoch_train_loss))
    val_losses.append(np.mean(epoch_val_loss))
    logger.info(f"Epoch {epoch+1}: \
          Train loss -- {losses[-1]:.3f}\t \
          Val loss -- {val_losses[-1]:.3f}")

    if es.step(torch.FloatTensor([val_losses[-1]])) and epoch > 20:
        break

ckpt_dir = f"{graph_dir}/budget_{budget}/encoder"
os.makedirs(ckpt_dir, exist_ok=True)
torch.save(encoder, f"{ckpt_dir}/encoder.pth")
