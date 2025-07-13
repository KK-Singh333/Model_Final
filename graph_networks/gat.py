import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_heads=8, num_layers=2, dropout=0.6):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # First GAT layer
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_channels, heads=num_heads, dropout=dropout))
        
        # Middle GAT layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        
        # Last GAT layer (uses single head attention for final representation)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout))
        
        # Output layer for binary classification
        self.classifier = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch=None):
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        from torch_geometric.nn import global_mean_pool
        if batch is not None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        else:
            # If no batch vector provided, assume single graph
            x = torch.mean(x, dim=0, keepdim=True)  # [1, hidden_channels]
        
        # Final classification
        x = self.classifier(x)
        
        # Return shape [batch_size]
        return torch.sigmoid(x).view(-1)
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_architecture_details(self):
        """Return a string with the detailed architecture"""
        details = []
        details.append(f"GAT Architecture:")
        details.append(f"- Input features: {self.num_node_features}")
        details.append(f"- Hidden channels: {self.hidden_channels}")
        details.append(f"- Attention heads: {self.num_heads}")
        details.append(f"- Number of layers: {self.num_layers}")
        details.append(f"- Dropout rate: {self.dropout}")
        
        details.append(f"\nLayer structure:")
        for i, conv in enumerate(self.convs):
            if i == 0:
                details.append(f"  Layer {i+1}: GATConv({self.num_node_features} -> {self.hidden_channels}, heads={self.num_heads})")
            elif i < self.num_layers - 1:
                details.append(f"  Layer {i+1}: GATConv({self.hidden_channels*self.num_heads} -> {self.hidden_channels}, heads={self.num_heads})")
            else:
                details.append(f"  Layer {i+1}: GATConv({self.hidden_channels*self.num_heads} -> {self.hidden_channels}, heads=1)")
        
        details.append(f"  Output Layer: Linear({self.hidden_channels} -> 1)")
        
        return "\n".join(details)
