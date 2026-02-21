# src/models/flexible_gnn.py
#
# Purpose:
#   A flexible, modular GNN model supporting:
#       - GCN
#       - GraphSAGE
#       - GAT
#       - GIN
#       - EGNN (geometry-aware)
#       - Custom MPNN layers
#
#   Works for centralized + federated learning.
#   Fully commented for clarity.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    global_mean_pool,
)

# Optional: EGNN (if installed)
try:
    from egnn_pytorch import EGNN
    HAS_EGNN = True
except ImportError:
    HAS_EGNN = False


# ---------------------------------------------------------
# Optional: Custom MPNN layer
# ---------------------------------------------------------
from torch_geometric.nn import MessagePassing

class CustomMPNN(MessagePassing):
    """
    Example custom MPNN layer.
    Replace message(), update() with your own logic.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_dim, out_dim)

    def message(self, x_j, edge_attr=None):
        return x_j

    def update(self, aggr_out):
        return self.lin(aggr_out)


# ---------------------------------------------------------
# Flexible GNN model
# ---------------------------------------------------------
class FlexibleGNN(nn.Module):
    """
    Flexible GNN supporting multiple architectures.

    Args:
        in_dim:     input feature dimension
        hidden_dim: hidden layer size
        out_dim:    output dimension (e.g., 2 for x,y)
        arch:       "gcn", "sage", "gat", "gin", "egnn", "custom"
        num_layers: number of GNN layers
        dropout:    dropout probability
        pooling:    graph-level output (optional)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        out_dim=2,
        arch="sage",
        num_layers=3,
        dropout=0.1,
        pooling=False,
    ):
        super().__init__()

        self.arch = arch
        self.dropout = dropout
        self.pooling = pooling
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(self._make_layer(in_dim, hidden_dim))

        # Middle layers
        for _ in range(num_layers - 1):
            self.layers.append(self._make_layer(hidden_dim, hidden_dim))

        # Output head
        self.head = nn.Linear(hidden_dim, out_dim)

    # ---------------------------------------------------------
    # Factory: create a layer based on architecture
    # ---------------------------------------------------------
    def _make_layer(self, in_dim, out_dim):
        if self.arch == "gcn":
            return GCNConv(in_dim, out_dim)

        elif self.arch == "sage":
            return SAGEConv(in_dim, out_dim)

        elif self.arch == "gat":
            return GATConv(in_dim, out_dim, heads=1, concat=False)

        elif self.arch == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)

        elif self.arch == "egnn":
            if not HAS_EGNN:
                raise ImportError("Install egnn-pytorch to use EGNN")
            return EGNN(
                dim=in_dim,
                edge_dim=1,
                m_dim=out_dim,
                fourier_features=0,
            )

        elif self.arch == "custom":
            return CustomMPNN(in_dim, out_dim)

        else:
            raise ValueError(f"Unknown GNN architecture: {self.arch}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # EGNN uses coordinates explicitly
        if self.arch == "egnn":
            pos = data.pos
            for layer in self.layers:
                x, pos = layer(x, pos, edge_index)
            node_emb = x

        else:
            # Standard MPNN layers
            for layer in self.layers:
                x = layer(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            node_emb = x

        # Graph-level output
        if self.pooling:
            graph_emb = global_mean_pool(node_emb, data.batch)
            return self.head(graph_emb)

        # Node-level output
        return self.head(node_emb)