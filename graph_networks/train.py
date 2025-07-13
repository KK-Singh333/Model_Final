import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Import local modules
from gat import GAT  # Updated import
from utils.dataset import load_data
# Import visualization utilities
from visualization import draw_gat_architecture  # Updated import

# Setup logging
def setup_logging(save_dir):
    log_file = os.path.join(save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch.x, batch.edge_index)
        
        # Ensure output and target have same dimensions [batch_size]
        target = batch.y.float().view(-1)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, device, loader, dataset_name="Validation"):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index)
            
            # Ensure consistent dimensions
            target = batch.y.float().view(-1)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend((output > 0.5).float().cpu().numpy())
            y_score.extend(output.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    # Log metrics
    metrics_str = f"{dataset_name} Results: "
    metrics_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logging.info(metrics_str)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='GAT for Binary Classification')
    parser.add_argument('--data-path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--filename', type=str, default='brain_graphs.csv', help='Dataset filename')
    parser.add_argument('--num-features', type=int, default=128, help='Number of node features')
    parser.add_argument('--hidden-channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model')
    parser.add_argument('--checkpoint-freq', type=int, default=25, help='Save checkpoint every N steps')
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Set up logging
    logger = setup_logging(args.save_dir)
    
    # Log training configuration
    logger.info(f"Training configuration: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    # Record start time
    start_time = time.time()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    train_loader, val_loader, test_loader = load_data(
        root=args.data_path,
        filename=args.filename,
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = GAT(
        num_features=args.num_features,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Count and display number of parameters
    param_count = model.count_parameters()
    logger.info(f"Model has {param_count:,} trainable parameters")
    
    # Log detailed model architecture
    logger.info(model.get_architecture_details())
    
    # Visualize model architecture
    architecture_path = os.path.join(args.save_dir, 'model_architecture.png')
    draw_gat_architecture(model, save_path=architecture_path)
    logger.info(f"Model architecture visualization saved to {architecture_path}")
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_auc = 0
    train_losses = []
    val_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_metric = evaluate(model, device, val_loader, "Validation")
        for k in val_metrics:
            val_metrics[k].append(val_metric[k])
        
        # Evaluate on test set periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            test_metric = evaluate(model, device, test_loader, "Test")
        
        # Save checkpoint every N steps
        if epoch % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metric,
                'best_val_auc': best_val_auc,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch}")
        
        # Save best model
        if val_metric['auc'] > best_val_auc:
            best_val_auc = val_metric['auc']
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with validation AUC: {best_val_auc:.4f}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        logger.info(f'Epoch: {epoch:03d}/{args.epochs}, Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, '
              f'Val AUC: {val_metric["auc"]:.4f}, Val F1: {val_metric["f1"]:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    for metric_name, metric_values in val_metrics.items():
        plt.plot(metric_values, label=metric_name)
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, 'training_curves.png')
    plt.savefig(plot_path)
    logger.info(f"Training curves saved to {plot_path}")
    
    # Test best model
    logger.info("Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
    test_metrics = evaluate(model, device, test_loader, "Final Test")
    
    # Save test results
    results_path = os.path.join(args.save_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Best model test results:\n")
        for metric_name, metric_value in test_metrics.items():
            f.write(f'{metric_name}: {metric_value:.4f}\n')
    
    # Calculate and log total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
