from models.sparse_graph import SparseGraphAttentionLayer
import torch
import torch.nn as nn

class MultiHeadSparseGAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.6, alpha=0.2, final_layer=False):
        super().__init__()
        self.final_layer = final_layer
        self.heads = nn.ModuleList([
            SparseGraphAttentionLayer(
                in_features,
                out_features,
                dropout=dropout,
                alpha=alpha,
                concat=not final_layer  # concat=True for hidden, False for final
            )
            for _ in range(num_heads)
        ])

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]

        if self.final_layer:
            # Average the outputs
            return torch.mean(torch.stack(head_outputs), dim=0)
        else:
            # Concatenate outputs
            return torch.cat(head_outputs, dim=1)
