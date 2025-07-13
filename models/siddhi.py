import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from layers.Embed import TokenChannelEmbedding
from models.gat import GAT
class MatrixFactorizationLayer(nn.Module):
    def __init__(self, input_dim_p, input_dim_c, latent_dim):
        super(MatrixFactorizationLayer, self).__init__()
        self.initialised=False
        self.W_patch = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.W_channel = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W_patch)
        nn.init.xavier_uniform_(self.W_channel)

    def forward(self, X_patch, X_channel):
        Z_patch = X_patch @ self.W_patch
        Z_channel = X_channel @ self.W_channel
        H_factor = F.relu(Z_patch @ Z_channel.transpose(1, 2))
        return H_factor



class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim_p,input_dim_c,latent_dim):
        super(CrossAttentionLayer, self).__init__()
        self.latent_dim=latent_dim
    def top_k_sparse_mask(self,scores,k):
      mean = scores.mean(dim=-1, keepdim=True)
      std = scores.std(dim=-1, keepdim=True)
      threshold = mean + std/16

      mask = torch.where(scores >= threshold, torch.tensor(0.0, device=scores.device), torch.tensor(float('-inf'), device=scores.device))
      # print(mask)
    #   mask = torch.zeros_like(scores, dtype=torch.float32)
      
      return mask
    def forward(self, X_patch, X_channel):
        # Patch to Channel
        A_patch_to_channel_attention_scores=X_patch @ X_channel.transpose(1, 2) / (self.latent_dim**0.5)
        A_patch_to_channel_attention_mask=self.top_k_sparse_mask(A_patch_to_channel_attention_scores,k=10)
        A_patch_to_channel_attention_scores+=A_patch_to_channel_attention_mask
        A_patch_to_channel = F.softmax(A_patch_to_channel_attention_scores, dim=-1)
        H_patch_to_channel = A_patch_to_channel @ X_channel
        
        # Channel to Patch
        A_channel_to_patch_attention_scores=X_channel @ X_patch.transpose(1, 2) / (self.latent_dim**0.5)
        A_channel_to_patch_attention_mask=self.top_k_sparse_mask(A_channel_to_patch_attention_scores,k=10)
        A_channel_to_patch_attention_scores+=A_channel_to_patch_attention_mask
        A_channel_to_patch = F.softmax(A_channel_to_patch_attention_scores, dim=-1)
        H_channel_to_patch = A_channel_to_patch @ X_patch
        
        # Hybrid representation
        H_hybrid = H_patch_to_channel @ H_channel_to_patch.transpose(1, 2)
        return H_hybrid


class WeightedFusionLayer(nn.Module):
    def __init__(self, input_dim_p, input_dim_c):
        super(WeightedFusionLayer, self).__init__()
        self.W_alpha = nn.Parameter(torch.empty(input_dim_c, input_dim_c))
        self.W_beta = nn.Parameter(torch.empty(input_dim_c, input_dim_c))
        self.W_out = nn.Parameter(torch.empty(input_dim_p, input_dim_c))
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, H_factor, H_hybrid):
        alpha = F.softmax(H_factor @ self.W_alpha, dim=-1)
        beta = F.softmax(H_hybrid @ self.W_beta, dim=-1)
        H_final = alpha * H_factor + beta * H_hybrid
        u = torch.sum(H_final * self.W_out, dim=(1, 2)) + self.bias
        y = torch.sigmoid(u)
       
        return y

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        if configs.no_temporal_block and configs.no_channel_block:
            raise ValueError("At least one of the two blocks should be True")
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        if configs.no_channel_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding=TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        input_dim_p=int((configs.d_model/patch_len_list[0]))+1
        input_dim_c=up_dim_list[0]
        latent_dim=configs.d_model
        self.matrix_factorization = MatrixFactorizationLayer(input_dim_p,input_dim_c, latent_dim)
        self.cross_attention = CrossAttentionLayer(input_dim_p,input_dim_c,latent_dim)
        self.weighted_fusion = WeightedFusionLayer(input_dim_p,input_dim_c)
        self.graph_layer=GAT(
        args=configs,
        num_node_features=128 if 'ADSZ' in configs.model_id else 256 ,
        hidden_channels=configs.d_model,
        num_heads=4,
        num_layers=2,
        dropout=0.2
        )
        self.channels=19
        self.project_nodes= nn.Parameter(torch.empty(configs.d_model, input_dim_c))
        
        # print((2*input_dim_p+self.channels)*input_dim_c)
        self.projection=torch.nn.Linear((2*input_dim_p+self.channels)*input_dim_c,configs.num_class,bias=True)
        # self.projection=torch.nn.Linear(4384,configs.num_class,bias=True) #Uncomment for APAVA
        # self.projection=torch.nn.Linear(5263,configs.num_class,bias=True)#Uncomment for ADFD
        # self.projection=torch.nn.Linear(,configs.num_class,bias=True)
        nn.init.xavier_uniform_(self.project_nodes)
        # nn.init.xavier_uniform_(self.projection)
    def drop_edges(self, edge_index, k=0.3):
        num_edges = edge_index.size(1)  # edge_index shape: [2, num_edges]
        keep_num = int(k * num_edges)

        # Generate a random permutation of indices
        perm = torch.randperm(num_edges)

        # Keep the first `keep_num` edges
        keep_idx = perm[:keep_num]

        # Return the filtered edge_index
        return edge_index[:, keep_idx]
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,graph_batch=None):
        # Matrix Factorization
        X_patch, X_channel = self.enc_embedding(x_enc)
        X_patch=X_patch[0]
        X_channel=X_channel[0]
        # print(X_channel.shape)
        H_factor = self.matrix_factorization(X_patch, X_channel)
        
        # Cross Attention
        H_hybrid = self.cross_attention(X_patch, X_channel)
        graph_batch.edge_index=self.drop_edges(graph_batch.edge_index)
        graph_matrix=self.graph_layer(graph_batch.x,graph_batch.edge_index,graph_batch.batch)
        projected_graph_matrix=torch.matmul(graph_matrix, self.project_nodes)
        print(H_factor.shape)
        print(H_hybrid.shape)
        print(projected_graph_matrix.shape)
        concatenated = torch.cat([H_factor, H_hybrid, projected_graph_matrix], dim=1)
        flattened = torch.flatten(concatenated, start_dim=1)

        y = self.projection(flattened)
        
        return y,H_factor,H_hybrid,projected_graph_matrix



