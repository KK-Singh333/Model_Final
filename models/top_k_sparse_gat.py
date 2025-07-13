import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

class SparseTopKAttentionGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, k=5, dropout=0.6, concat=True, activation=F.elu):
        super().__init__()
        self.k = k
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.concat = concat
        self.activation = activation

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        N = x.size(0)
        H, C = self.heads, self.out_channels

        x = self.lin(x).view(N, H, C)  # [N, H, C]

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        row, col = edge_index

        x_i = x[row]  # target node features [E, H, C]
        x_j = x[col]  # source node features [E, H, C]

        # Attention score computation
        alpha_input = torch.cat([x_i, x_j], dim=-1)  # [E, H, 2C]
        e = (alpha_input * self.att).sum(dim=-1)     # [E, H]
        e = F.leaky_relu(e, negative_slope=0.2)

        # Top-k masking
        topk_mask = self._topk_edge_mask(e, row, self.k)
        e = e.masked_fill(~topk_mask, float('-inf'))

        # Normalize & dropout
        alpha = softmax(e, row)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing
        out = alpha.unsqueeze(-1) * x_j                # [E, H, C]
        out = scatter_add(out, row, dim=0, dim_size=N) # [N, H, C]

        # Combine heads
        if self.concat:
            out = out.view(N, H * C)  # [N, H*C]
        else:
            out = out.mean(dim=1)    # [N, C]

        # Optional activation
        if self.activation is not None:
            out = self.activation(out)

        return out

    def _topk_edge_mask(self, scores, row, k):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for h in range(scores.size(1)):
            scores_h = scores[:, h]
            unique_nodes = torch.unique(row)
            for node in unique_nodes:
                node_mask = (row == node)
                if node_mask.sum() <= k:
                    mask[node_mask, h] = True
                else:
                    topk = torch.topk(scores_h[node_mask], k).indices
                    idx = torch.nonzero(node_mask, as_tuple=False).squeeze()[topk]
                    mask[idx, h] = True
        return mask
