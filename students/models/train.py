import argparse
import random
import torch as th
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from torch.optim import AdamW
from transformers.optimization import (
    get_polynomial_decay_schedule_with_warmup,
)

import os
from pathlib import Path

'''
Classical GNN
GAT
'''
import dgl
from classical_gnn import ClassicGNN

def train_classical_gnn(model, optimizer, data_loader, lr_scheduler):
    model.train()
    epoch_loss = 0

    loss_fn = nn.L1Loss()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    for ( 
        batched_graph,
        labels,
        num_nodes_per_graph
    ) in data_loader: 
        optimizer.zero_grad()
        
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

        # 添加这段验证代码 - 只打印前一个批次的少量样本
        if not hasattr(train_classical_gnn, "debug_printed"):
            train_classical_gnn.debug_printed = True
            
            print("\n==== 验证标签错位问题 ====")
            
            # 从第一个批次取少量样本进行分析
            sample_size = min(3, len(valid_labels))
            
            print("当前使用的标签顺序(未排序):")
            print(valid_labels[:sample_size].detach().cpu().numpy())
            
            print("\n预测值顺序(对应MLPPredictionBlock内部排序后):")
            print(valid_scores[:sample_size].detach().cpu().numpy())
            
            # 验证MLPPredictionBlock内部确实进行了排序
            print("\n让我们查看如果按标签排序后的结果:")
            sample_labels = valid_labels[:sample_size].detach().cpu()
            sorted_indices = th.argsort(sample_labels)
            sorted_labels = sample_labels[sorted_indices]
            print("排序后标签:", sorted_labels.numpy())
            
            print("\n这表明预测值可能是按照排序后的标签顺序输出的")
            print("而计算损失时使用的是未排序的标签！")
            print("==== 验证结束 ====\n")

        loss = loss_fn(valid_scores, valid_labels)


        loss.backward()
        optimizer.step() 
        lr_scheduler.step()
        epoch_loss += loss.item() 


        
        del (
            batched_graph,
            labels,
            num_nodes_per_graph,
            batch_scores,
            loss,

        ) 
        th.cuda.empty_cache()

    epoch_loss /= len(data_loader)
    return epoch_loss


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

    return batched_graph, labels, num_nodes_per_graph

def train_val_classical_gnn(params):
    dataset = dgl.data.CSVDataset(params.dataset_path) 
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_data, val_data = th.utils.data.random_split(dataset, [train_size, val_size]) 

    
    train_loader = GraphDataLoader( 
        train_data, 
        batch_size=params.batch_size,
        shuffle=True, 
        collate_fn=collate_classical_gnn, 

    )
    val_loader = GraphDataLoader(
        val_data,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=collate_classical_gnn,

    )

    model = ClassicGNN(model_type=params.classicGNN_type) 
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    

    

    num_epochs = params.num_epochs
    total_updates = len(train_data) * num_epochs / params.batch_size
    
    warmup_updates = total_updates * 0.16

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup( 
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
        lr_end=1e-9,
        power=1.0,
    )

    epoch_train_MSEs, epoch_val_MSEs = [], [] 

    model_save_path = Path("model_weights")
    model_save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_path / ("best_model_" + str(params.classicGNN_type) + ".pth") 

    best_val_loss = float('inf')


    for epoch in range(num_epochs):
        
        epoch_train_loss = train_classical_gnn(model, optimizer, train_loader, lr_scheduler)
        
        
        epoch_val_loss = evaluate_classical_gnn(model, val_loader)
        
        
        epoch_train_MSEs.append(epoch_train_loss)
        epoch_val_MSEs.append(epoch_val_loss)
        
        
        print(f"Epoch={epoch + 1} | train_loss={epoch_train_loss:.3f} | val_loss={epoch_val_loss:.3f}")
        
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            th.save(model.state_dict(), best_model_path)

    index = epoch_val_MSEs.index(min(epoch_val_MSEs))

    val_mse = epoch_val_MSEs[index] 
    train_mse = epoch_train_MSEs[index]

    print("Val MSE: {:.4f}".format(val_mse))
    print("Train MSE: {:.4f}".format(train_mse))
    print("Best epoch index: {:.4f}".format(index)) 
    return val_mse, train_mse, index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default='data/ffw_distill', 
        type=str,
        help="Please give the path for the dataset",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Please give a value for random seed",
    )
    parser.add_argument(
        "--batch_size",
        default=128, 
        type=int,
        help="Please give a value for batch_size",
    )
    parser.add_argument(
        "--num_epochs",
        default=1600,
        type=int,
        help="Please give a value for num_epochs",
    )
    parser.add_argument(
        "--classicGNN_type",
        default='EdgeGAT',
        type=str,
        help="Please give a value for classicGNN_type",
    )
    args = parser.parse_args()

    
    random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)

    val_loss_deepgat, train_loss_deepgat, index_deepgat = train_val_classical_gnn(args)





