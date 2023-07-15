import os
import argparse
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from networks import CustomDataset
from networks import ClassifierAutoencoder
import utils



args = argparse.ArgumentParser()
args.add_argument("-d", '--dataset', required=True)
args.add_argument("-m", '--weight_model', default='TV', help='weight model')
args.add_argument("-k", '--budget', default=100)
args.add_argument("--gpu", type=int, default=0)
args.add_argument("--num_layers", type=int, default=1)
args.add_argument("--num_classes", type=int, default=4)
args.add_argument("--input_size", type=int, default=128)
args.add_argument("-b", "--batch_size", type=int, default=128)
opt = args.parse_args()

graph_name = opt.dataset
weight_model = opt.weight_model
budget = opt.budget
num_layers = opt.num_layers
num_classes = opt.num_classes
input_size = opt.input_size
batch_size = opt.batch_size

utils.set_seed(seed=0)

base_dir = os.path.join("data", graph_name, weight_model, "train", f"budget_{budget}", "encoder")
with open(os.path.join(base_dir, "train_data"), mode="rb") as f:
    data, labels = pickle.load(f)

dataset = CustomDataset(data, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
loss_fn = nn.MSELoss()

net = ClassifierAutoencoder(input_size, num_layers)
optimiser = torch.optim.Adam(net.parameters())

for epoch in range(200):
    for x, y in loader:
        optimiser.zero_grad()
        loss = loss_fn(net(x), x)
        loss.backward()
        optimiser.step()

with torch.no_grad():
    output = net(data, True)
    output = output.numpy()
    labels = labels.numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

colours = ["red", "blue", "black", "yellow", "green"]
for class_, colour in zip(range(num_classes), colours):
    ax1.scatter(output[labels == class_, 0], output[labels == class_, 1], color=colour, label=f"class {class_ + 1}")
ax1.legend()
ax1.title.set_text("embedding space")

good_labels = [i for i in range(data.shape[0]) if int(labels[i]) == 0]
good_point = torch.mean(data[good_labels], dim=0)
similarities = torch.norm((good_point.reshape((1, input_size)) - data), p=2, dim=1).numpy()

im1 = ax2.scatter(output[:, 0], output[:, 1], c=similarities)
fig.colorbar(im1, ax=ax2)
ax2.title.set_text("L2 distance")
plt.savefig(f"{base_dir}/embeddings.pdf")
plt.show()
