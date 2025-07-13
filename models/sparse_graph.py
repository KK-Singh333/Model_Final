import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_src = nn.Parameter(torch.empty(out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # x: (N, in_features)
        # edge_index: (2, E) — COO format, source and destination node indices

        N = x.size(0)
        E = edge_index.size(1)

        # Linear transformation
        Wh = torch.mm(x, self.W)  # (N, out_features)

        # Source and target node features for each edge
        src = edge_index[0]
        dst = edge_index[1]
        Wh_src = Wh[src]  # (E, out_features)
        Wh_dst = Wh[dst]  # (E, out_features)

        # Compute attention coefficients for each edge
        e = self.leakyrelu(
            (Wh_src @ self.a_src) + (Wh_dst @ self.a_dst)
        ).squeeze()  # (E,)

        # Normalize via softmax over incoming edges to each node
        e = torch.exp(e - torch.max(e))  # stability
        zero = torch.zeros(N, device=x.device)
        denom = torch.zeros(N, device=x.device).index_add_(0, dst, e) + 1e-16
        alpha = e / denom[dst]  # (E,)

        alpha = self.dropout(alpha)

        # Weighted sum of neighbor features
        out = torch.zeros_like(Wh)
        out = out.index_add_(0, dst, Wh_src * alpha.unsqueeze(-1))

        return F.elu(out) if self.concat else out
