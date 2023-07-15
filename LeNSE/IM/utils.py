import logging
import os
import os.path as osp
import time
import yaml
import random
import torch
import numpy as np

def get_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger

def write_weighted_graph_in_tmp_dir(graph, path):
    with open(path, 'w') as f:
        for u, v in graph.edges():
            weight = graph[u][v]['weight']
            if type(weight) == dict:
                weight = weight['weight']
            f.write(f"{u} {v} {weight}\n")


def write_tmp_file(graph, graph_dir):
    timestamp = str(round(time.time()))
    graph_tmp_dir = osp.join(graph_dir, "tmp")
    graph_path = osp.join(graph_tmp_dir, f"{timestamp}.txt")
    attr_path = osp.join(graph_tmp_dir, "attribute.txt")
    os.makedirs(graph_tmp_dir, exist_ok=True)
    write_weighted_graph_in_tmp_dir(graph, graph_path)
    with open(attr_path, "w") as f:
        f.write(f"n={graph.number_of_nodes()}\nm={graph.number_of_edges()}")
    return graph_tmp_dir, graph_path

def calculate_influence_spread(graph_path, seeds, size=1e5):
    # create soft link first
    EVALUATION_PROGRAM = osp.join(osp.dirname(__file__), "IMEvaluation", "evaluate")
    # tmp file
    tmp_dir = osp.join(osp.dirname(graph_path), "metric_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    stamp = str(round(time.time()))
    output_tmp_file = osp.join(tmp_dir, f"coverage_{stamp}.txt")
    seed_tmp_file = osp.join(tmp_dir, f"seed_{stamp}.txt")
    with open(seed_tmp_file, "w") as f:
        for seed in seeds:
            f.write(f"{seed}\n")
            
    command = f"{EVALUATION_PROGRAM} -seedFile {seed_tmp_file} -output {output_tmp_file} \
        -graphFile {graph_path} -size {str(int(size))}"
    res = os.system(command)
    assert res==0, "Evaluate influence spread error!!"
    coverage = float(open(output_tmp_file).read().strip())
    return coverage

def load_global_config():
    f = open(osp.join(osp.dirname(__file__), "config.yaml")).read()
    config = yaml.safe_load(f)
    return config

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False