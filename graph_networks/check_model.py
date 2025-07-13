import torch
import argparse
import os
import sys

# Add parent directory to the path so we can import from data_provider
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gat import GAT
from visualization import draw_gat_architecture
from eeg_graph_dataset import EEGGraphDataset
from data_provider.data_loader import ADSZIndependentLoader


class Args:
    def __init__(self):
        self.no_normalize = False


def main():
    parser = argparse.ArgumentParser(description='Check GAT Model Parameters')
    parser.add_argument('--data-path', type=str, default='./dataset/ADSZ', help='Path to dataset')
    parser.add_argument('--hidden-channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--save-dir', type=str, default='model_check', help='Directory to save outputs')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load a small portion of data to get dimensions
    print("Loading a sample of data to determine dimensions...")
    loader_args = Args()
    train_data = ADSZIndependentLoader(loader_args, args.data_path, flag='TRAIN')
    
    print(f"Data shape: {train_data.X.shape}, labels shape: {train_data.y.shape}")
    
    # Create a sample dataset and determine number of features
    dataset = EEGGraphDataset(train_data.X[:1], train_data.y[:1])
    sample = dataset[0]
    num_features = sample.x.shape[1]
    
    print(f"Number of node features: {num_features}")
    print(f"Number of nodes per graph: {sample.x.shape[0]}")
    print(f"Edge index shape: {sample.edge_index.shape}")
    
    # Initialize model
    model = GAT(
        num_node_features=num_features,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Print architecture details
    print("\n" + model.get_architecture_details())
    
    # Count parameters
    param_count = model.count_parameters()
    print(f"\nModel has {param_count:,} trainable parameters")
    
    # Display breakdown of parameters by layer
    print("\nParameter breakdown by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
    
    # Visualize architecture
    architecture_path = os.path.join(args.save_dir, 'model_architecture.png')
    draw_gat_architecture(model, save_path=architecture_path)
    print(f"\nModel architecture visualization saved to {architecture_path}")
    
    # Calculate memory requirements (approximate)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    # Assume 4 bytes per parameter for optimizer state (momentum and adaptive)
    optimizer_size = param_count * 4
    
    total_size = param_size + buffer_size + optimizer_size
    
    print(f"\nApproximate memory requirements:")
    print(f"Parameters: {param_size / 1024**2:.2f} MB")
    print(f"Buffers: {buffer_size / 1024**2:.2f} MB")
    print(f"Optimizer state: {optimizer_size / 1024**2:.2f} MB")
    print(f"Total: {total_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
