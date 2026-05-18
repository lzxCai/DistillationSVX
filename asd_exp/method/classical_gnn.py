import dgl
import torch
# import dgl.transforms as T
import dgl.data
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

'''
GCN
'''
class GCN(nn.Module):
    def __init__(
            self,
            in_dim=3,
            hidden_dim=512,
            num_encoder_layers=8
        ):
        super().__init__()

        # self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads)
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                dglnn.GraphConv(
                    hidden_dim,
                    hidden_dim,
                    norm='both',
                    weight=True,
                    bias=True
                )
                for _ in range(num_encoder_layers) # 直接stack多层即可
            ]
        )

    def forward(self, graph, inputs):
        # h = self.conv1(graph, inputs)
        h = self.node_encoder(inputs)
        for layer in self.layers: # Encoder层
            h_prime = layer(
                graph,
                h
            )
            if h.shape == h_prime.shape:
                h = h + h_prime
            else:
                raise ValueError("Shape mismatch in residual connection.")
            h = F.relu(h)
        return h

'''
GAT
'''
class GAT(nn.Module):
    def __init__(
            self,
            in_dim=3,
            hidden_dim=512,
            num_heads=8,
            num_encoder_layers=8
        ):
        super().__init__()

        # self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads)
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                dglnn.GATConv(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    residual=True
                )
                for _ in range(num_encoder_layers) # 直接stack多层即可
            ]
        )

    def forward(self, graph, inputs):
        # h = self.conv1(graph, inputs)
        h = self.node_encoder(inputs)
        for layer in self.layers: # Encoder层
            h = layer(
                graph,
                h
            )
            h = F.relu(h.mean(dim=1))
        return h

'''
EdgeGAT
'''
class EdgeGAT(nn.Module):
    def __init__(
            self,
            node_in_dim=3,
            node_hidden_dim=512,
            edge_in_dim=5,
            edge_hidden_dim=512,
            num_heads=8,
            num_encoder_layers=8
        ):
        super().__init__()

        # self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads)
        self.node_encoder = nn.Linear(node_in_dim, node_hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, edge_hidden_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                dglnn.EdgeGATConv(
                    node_hidden_dim,
                    edge_hidden_dim,
                    node_hidden_dim,
                    num_heads,
                    residual=True
                )
                for _ in range(num_encoder_layers) # 直接stack多层即可
            ]
        )

    def forward(self, graph, node_inputs, edge_inputs):
        # h = self.conv1(graph, inputs)
        h = self.node_encoder(node_inputs)
        e = self.edge_encoder(edge_inputs)
        for layer in self.layers: # Encoder层
            h = layer(
                graph,
                h,
                e
            )
            h = F.relu(h.mean(dim=1))
        return h

'''
GINE
'''
class GINE(nn.Module):
    def __init__(
            self,
            node_in_dim=3,
            node_hidden_dim=512,
            edge_in_dim=5,
            edge_hidden_dim=512,
            num_heads=8,
            num_encoder_layers=8
        ):
        super().__init__()

        # self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads)
        self.node_encoder = nn.Linear(node_in_dim, node_hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, edge_hidden_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                dglnn.GINEConv(
                    nn.Linear(node_hidden_dim, node_hidden_dim)
                )
                for _ in range(num_encoder_layers) # 直接stack多层即可
            ]
        )

    def forward(self, graph, node_inputs, edge_inputs):
        # h = self.conv1(graph, inputs)
        h = self.node_encoder(node_inputs)
        e = self.edge_encoder(edge_inputs)
        for layer in self.layers: # Encoder层
            h_prime = layer(
                graph,
                h,
                e
            )
            if h.shape == h_prime.shape:
                h = h + h_prime
            else:
                raise ValueError("Shape mismatch in residual connection.")
            h = F.relu(h)
        return h

'''
GatedGCN
'''
class GatedGCN(nn.Module):
    def __init__(
            self,
            node_in_dim=3,
            node_hidden_dim=512,
            edge_in_dim=5,
            edge_hidden_dim=512,
            num_heads=8,
            num_encoder_layers=8
        ):
        super().__init__()

        # self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads)
        self.node_encoder = nn.Linear(node_in_dim, node_hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, edge_hidden_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                dglnn.GatedGCNConv(
                    node_hidden_dim,
                    edge_hidden_dim,
                    node_hidden_dim,
                    dropout=0.1,
                    batch_norm=True,
                    activation=F.relu
                )
                for _ in range(num_encoder_layers) # 直接stack多层即可
            ]
        )

    def forward(self, graph, node_inputs, edge_inputs):
        # h = self.conv1(graph, inputs)
        h = self.node_encoder(node_inputs)
        e = self.edge_encoder(edge_inputs)
        for layer in self.layers: # Encoder层
            h, e = layer(
                graph,
                h,
                e
            )
        return h

'''
Shared Prediction Block
'''
class MLPPredictionBlock(nn.Module):
    def __init__(self,
                 in_feat=512
        ):
        super(MLPPredictionBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, int(in_feat // 2)),
            nn.ReLU(),
            nn.Linear(int(in_feat / 2), int(in_feat / 4)),
            nn.ReLU(),
            nn.Linear(int(in_feat / 4), 1)
        )
    
    def forward(self, h, node_label, num_nodes_per_graph):
        results = []
        
        num_graphs = len(num_nodes_per_graph)
        start_index = 0
        
        for i in range(num_graphs):
            end_index = start_index + num_nodes_per_graph[i]
            
            valid_indices = node_label[start_index:end_index] != -1
            valid_x = h[start_index:end_index][valid_indices]
            valid_labels = node_label[start_index:end_index][valid_indices]
            
            sorted_indices = torch.argsort(valid_labels)
            sorted_x = valid_x[sorted_indices]

            predicted_values = self.mlp(sorted_x).squeeze(-1)
            
            results.append(predicted_values)

            start_index = end_index

        final_results = pad_sequence(results, batch_first=True, padding_value=-1)
        
        return final_results

    
class ClassicGNN(nn.Module):
    def __init__(
            self,
            model_type
        ):
        super().__init__()
        self.model_type = model_type
        if model_type == 'GCN':
            self.model = GCN()
        elif model_type == 'GAT':
            self.model = GAT()
        elif model_type == 'EdgeGAT':
            self.model = EdgeGAT()
        elif model_type == 'GINE':
            self.model = GINE()
        elif model_type == 'GatedGCN':
            self.model = GatedGCN()
        
        self.pred = MLPPredictionBlock()

    def forward(self, g, num_nodes_per_graph):
        h = g.ndata['feat']
        e = g.edata["feat"]
        labels = g.ndata['label']

        if self.model_type  == 'GCN' or self.model_type  == 'GAT':
            h = self.model(g, h) # (nums_node_in_batch, hidden_dim)
        elif self.model_type  == 'EdgeGAT' or self.model_type  == 'GINE' or self.model_type  == 'GatedGCN':
            h = self.model(g, h, e)
        
        h = self.pred(h, labels, num_nodes_per_graph) # (batch_size, nums_unique_label)
        return h