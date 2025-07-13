import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def create_adjacency_matrix(num_nodes, connectivity_type='full'):
    """
    Create an adjacency matrix for the EEG electrodes
    Args:
        num_nodes: number of EEG electrodes/channels
        connectivity_type: type of connectivity ('full', 'neighbors', etc.)
    Returns:
        edge_index: tensor of shape [2, num_edges] containing the indices of connected nodes
    """
    if connectivity_type == 'full':
        # Fully connected graph
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return edge_index


class EEGGraphDataset(Dataset):
    def __init__(self, eeg_data, labels):
        """
        Dataset for EEG data represented as graphs
        Args:
            eeg_data: numpy array of shape [num_samples, seq_len, num_channels]
            labels: numpy array of shape [num_samples]
        """
        self.eeg_data = eeg_data
        self.labels = labels
        self.num_samples = eeg_data.shape[0]
        self.seq_len = eeg_data.shape[1]
        self.num_channels = eeg_data.shape[2]
        
        # Create edge index (graph connectivity)
        self.edge_index = create_adjacency_matrix(self.num_channels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get EEG time series data for one sample
        eeg_sample = self.eeg_data[idx]  # [seq_len, num_channels]
        # print(eeg_sample.shape)
        # Use the time series directly as node features
        # Each node (channel) has seq_len features
        node_features = eeg_sample.transpose()  # [num_channels, seq_len]
        # node_features=eeg_sample
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create the graph with label as a single float value
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        
        data = Data(x=x, edge_index=self.edge_index, y=y)
            
        return data


def load_eeg_graph_data(eeg_data, labels, batch_size=32):
    """
    Load EEG data as graphs for GAT model
    Args:
        eeg_data: dict with keys 'train', 'val', 'test', each containing numpy array of shape [num_samples, seq_len, num_channels]
        labels: dict with keys 'train', 'val', 'test', each containing numpy array of shape [num_samples]
        batch_size: batch size for dataloader
    Returns:
        train_loader, val_loader, test_loader: dataloaders for train, validation and test sets
    """
    train_dataset = EEGGraphDataset(eeg_data['train'], labels['train'])
    val_dataset = EEGGraphDataset(eeg_data['val'], labels['val'])
    test_dataset = EEGGraphDataset(eeg_data['test'], labels['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
