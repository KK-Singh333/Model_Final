import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def draw_gat_architecture(model, save_path=None):
    """
    Draw the GAT model architecture and save as a PNG file
    Args:
        model: GAT model instance
        save_path: path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get model details
    num_node_features = model.num_node_features
    hidden_channels = model.hidden_channels
    num_heads = model.num_heads
    num_layers = model.num_layers
    
    # Colors
    colors = {
        'input': '#66c2a5',
        'gat': '#fc8d62',
        'attn_head': '#8da0cb',
        'output': '#e78ac3',
        'text': '#000000'
    }
    
    # Define coordinates
    max_width = 12
    max_height = 10
    margin = 1
    x_start = margin
    x_end = max_width - margin
    y_start = margin
    y_end = max_height - margin
    
    # Draw title
    ax.text(max_width/2, y_end + 0.5, f'Graph Attention Network (GAT) Architecture',
            horizontalalignment='center', fontsize=14, fontweight='bold')
    
    # Draw input layer
    input_height = 2
    input_rect = patches.Rectangle((x_start, y_start), 2, input_height, 
                                  linewidth=1, edgecolor='black', facecolor=colors['input'], alpha=0.7)
    ax.add_patch(input_rect)
    ax.text(x_start + 1, y_start + input_height/2, f'Input\n({num_node_features})', 
            horizontalalignment='center', verticalalignment='center')
    
    # Calculate spacing for GAT layers
    gat_width = 2
    gat_spacing = (x_end - x_start - (num_layers * gat_width)) / (num_layers + 1)
    
    # Draw GAT layers
    for i in range(num_layers):
        x_pos = x_start + gat_spacing * (i + 1) + gat_width * i
        
        # GAT layer box
        gat_rect = patches.Rectangle((x_pos, y_start), gat_width, 3, 
                                     linewidth=1, edgecolor='black', facecolor=colors['gat'], alpha=0.7)
        ax.add_patch(gat_rect)
        
        # Layer name
        if i == 0:
            in_features = num_node_features
            heads = num_heads
        elif i < num_layers - 1:
            in_features = hidden_channels * num_heads
            heads = num_heads
        else:
            in_features = hidden_channels * num_heads
            heads = 1
            
        out_features = hidden_channels
        
        ax.text(x_pos + gat_width/2, y_start + 1.5, f'GAT Layer {i+1}\n({in_features} → {out_features})\nHeads: {heads}', 
                horizontalalignment='center', verticalalignment='center')
        
        # Draw attention heads for the first layer (to illustrate)
        if i == 0:
            head_height = 0.4
            head_spacing = 0.1
            total_head_height = heads * head_height + (heads - 1) * head_spacing
            head_start_y = y_start + 4
            
            for h in range(min(heads, 5)):  # Draw up to 5 heads to save space
                head_y = head_start_y + h * (head_height + head_spacing)
                head_rect = patches.Rectangle((x_pos, head_y), gat_width, head_height, 
                                             linewidth=1, edgecolor='black', facecolor=colors['attn_head'], alpha=0.7)
                ax.add_patch(head_rect)
                ax.text(x_pos + gat_width/2, head_y + head_height/2, f'Head {h+1}', 
                        horizontalalignment='center', verticalalignment='center', fontsize=8)
            
            # If there are more heads, indicate with ellipsis
            if heads > 5:
                ax.text(x_pos + gat_width/2, head_start_y + 5 * (head_height + head_spacing), "...", 
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
    
    # Draw output layer
    output_rect = patches.Rectangle((x_end - 2, y_start), 2, input_height, 
                                   linewidth=1, edgecolor='black', facecolor=colors['output'], alpha=0.7)
    ax.add_patch(output_rect)
    ax.text(x_end - 1, y_start + input_height/2, f'Output\n(1)', 
            horizontalalignment='center', verticalalignment='center')
    
    # Draw arrows connecting layers
    arrow_props = dict(arrowstyle='->', linewidth=1.5, color='gray')
    
    # Input to first GAT
    first_gat_x = x_start + gat_spacing + gat_width/2
    ax.annotate('', xy=(first_gat_x - gat_width/2, y_start + input_height/2), 
                xytext=(x_start + 2, y_start + input_height/2), arrowprops=arrow_props)
    
    # Between GAT layers
    for i in range(num_layers - 1):
        start_x = x_start + gat_spacing * (i + 1) + gat_width * (i + 1)
        end_x = x_start + gat_spacing * (i + 2) + gat_width * (i + 1)
        ax.annotate('', xy=(end_x, y_start + 1.5), xytext=(start_x, y_start + 1.5), arrowprops=arrow_props)
    
    # Last GAT to output
    last_gat_x = x_start + gat_spacing * num_layers + gat_width * (num_layers - 1)
    ax.annotate('', xy=(x_end - 2, y_start + input_height/2), 
                xytext=(last_gat_x + gat_width, y_start + 1.5), arrowprops=arrow_props)
    
    # Add pooling layer indicator
    pooling_x = (last_gat_x + gat_width + (x_end - 2)) / 2
    ax.text(pooling_x, y_start + input_height/2 - 0.5, "Global Mean\nPooling", 
            horizontalalignment='center', verticalalignment='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # Add model stats
    param_count = model.count_parameters()
    stats_text = f"Total parameters: {param_count:,}"
    ax.text(max_width/2, 0.3, stats_text, horizontalalignment='center', fontsize=12)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, max_width)
    ax.set_ylim(0, max_height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture visualization saved to {save_path}")
    
    return fig, ax
