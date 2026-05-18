import argparse
import random
import torch as th
import torch.nn as nn
# from accelerate import Accelerator
# from dataset_our_transfer import NetworkedWCDDataset
from dgl.data import download
from dgl.dataloading import GraphDataLoader
from classical_gnn import ClassicGNN
# from ogb.graphproppred import Evaluator
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)

import dgl
import os
from pathlib import Path
import time
import numpy as np

def evaluate_classical_gnn(model, data_loader):
    model.eval()
    epoch_loss = 0
    loss_fn = nn.L1Loss()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    with th.no_grad():

        for (
        batched_graph,
        labels,
        num_nodes_per_graph
        ) in data_loader:

            device = th.device('cuda' if th.cuda.is_available() else 'cpu')

            batched_graph = batched_graph.to(device)
            batched_graph = dgl.add_self_loop(batched_graph)

            num_nodes_per_graph = num_nodes_per_graph.to(device)
            labels = [label.to(device) for label in labels]
            batch_scores = model(batched_graph, num_nodes_per_graph).float()

            max_unique_labels_in_batch = batch_scores.shape[1]
            cropped_labels = th.stack([label[:max_unique_labels_in_batch] for label in labels])
            

            mask = cropped_labels != -1
            valid_labels = cropped_labels[mask]
            valid_scores = batch_scores[mask]

            loss = loss_fn(valid_scores, valid_labels)

            epoch_loss += loss.item()


        epoch_loss /= len(data_loader)

    return epoch_loss

def collate_classical_gnn(batch):
    graphs, labels = zip(*batch)

    batched_graph = dgl.batch(graphs)

    num_nodes_per_graph = th.tensor(batched_graph.batch_num_nodes())
    
    labels = labels

    return batched_graph, labels, num_nodes_per_graph

def train_val_pipeline(params):
    dataset = dgl.data.CSVDataset(params.dataset_path) 
    
    # print(f"Number of graphs: {len(dataset)}")
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # Load pre-trained model.
    model = ClassicGNN()
    model = model.to(device)
    state_dict = th.load("method/model_weights.pth") # 【填写你要验证的模型】
    model.load_state_dict(state_dict)

    full_dataset_loader = GraphDataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
    )
    
    full_dataset_mae = evaluate_classical_gnn(model, full_dataset_loader)
    
    print(f"MAE on the full dataset: {full_dataset_mae:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default='data/ffw_origin', # 【填你的数据集路径：ffw_origin 或者 nffw_origin】
        type=str,
        help="Please give the path for the dataset",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Please give a value for batch_size",
    )

    args = parser.parse_args()

    train_val_pipeline(args)