# Attention Visualization Tool

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from contextlib import contextmanager

class AttentionCaptureManager:
    """Manager to capture attention weights during model forward pass"""
    
    def __init__(self, model, layer_idx, head_idx=None):
        """
        Args:
            model: GPT model instance
            layer_idx: Which transformer layer to capture attention from (0-indexed)
            head_idx: Which attention head to capture (0-indexed). If None, captures all heads
        """
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.attention_weights = None
        self.hook = None
        
    def __enter__(self):
        # Register hook on the specified layer's attention module
        target_layer = self.model.transformer.h[self.layer_idx].attn
        self.hook = target_layer.register_forward_hook(self._capture_attention)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()
            
    def _capture_attention(self, module, input, output):
        """Hook function to capture attention weights"""
        # Check if this is a shadow audio model (returns tuple) or regular model
        if isinstance(output, tuple):
            # Shadow audio model returns (y, audio_mixed)
            # We need to access the attention weights that were computed
            # Unfortunately, they're not directly returned, so we need to modify our approach
            pass
        else:
            # Regular model - we need to access attention weights
            # This requires modification of the forward pass to expose them
            pass
            
    def get_attention_weights(self):
        """Get the captured attention weights"""
        return self.attention_weights


class AttentionHook:
    """Hook to capture attention weights from CausalSelfAttention layers"""
    
    def __init__(self):
        self.attention_weights = None
        
    def __call__(self, module, input, output):
        # For shadow audio models, we need to capture from the manual attention computation
        # Since the current model doesn't expose attention weights directly, 
        # we'll need to modify the forward method temporarily
        pass


def modify_attention_for_capture(model, layer_idx):
    """Temporarily modify the attention layer to capture weights"""
    target_layer = model.transformer.h[layer_idx].attn
    original_forward = target_layer.forward
    captured_attention = {'weights': None}
    
    def forward_with_capture(*args, **kwargs):
        if hasattr(target_layer, '_forward_with_audio'):
            # Shadow audio model
            x, audio_norm = args[0], args[1]
            B, T, C = x.size()
            
            # Replicate the attention computation to capture weights
            q, k, v = target_layer.c_attn(x).split(target_layer.n_embd, dim=2)
            k = k.view(B, T, target_layer.n_head, C // target_layer.n_head).transpose(1, 2)
            q = q.view(B, T, target_layer.n_head, C // target_layer.n_head).transpose(1, 2)
            v = v.view(B, T, target_layer.n_head, C // target_layer.n_head).transpose(1, 2)
            
            # Compute attention weights
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
            att = att.masked_fill(target_layer.bias[:,:,:T,:T] == 0, float('-inf'))
            att = target_layer.my_softmax(att)
            
            # Capture the attention weights
            captured_attention['weights'] = att.detach().clone()
            
            # Continue with normal forward pass
            return original_forward(*args, **kwargs)
        else:
            # Regular model
            x = args[0]
            B, T, C = x.size()
            
            # Replicate attention computation
            q, k, v = target_layer.c_attn(x).split(target_layer.n_embd, dim=2)
            k = k.view(B, T, target_layer.n_head, C // target_layer.n_head).transpose(1, 2)
            q = q.view(B, T, target_layer.n_head, C // target_layer.n_head).transpose(1, 2)
            v = v.view(B, T, target_layer.n_head, C // target_layer.n_head).transpose(1, 2)
            
            if target_layer.flash:
                # Can't capture from flash attention easily
                print("Warning: Cannot capture attention weights from flash attention")
                return original_forward(*args, **kwargs)
            else:
                # Manual attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
                att = att.masked_fill(target_layer.bias[:,:,:T,:T] == 0, float('-inf'))
                att = target_layer.my_softmax(att)
                
                # Capture the attention weights
                captured_attention['weights'] = att.detach().clone()
                
                # Continue with normal forward pass
                return original_forward(*args, **kwargs)
    
    # Temporarily replace the forward method
    target_layer.forward = forward_with_capture
    
    return captured_attention, original_forward


@contextmanager
def capture_attention_weights(model, layer_idx):
    """Context manager to capture attention weights from a specific layer"""
    captured_data, original_forward = modify_attention_for_capture(model, layer_idx)
    try:
        yield captured_data
    finally:
        # Restore original forward method
        model.transformer.h[layer_idx].attn.forward = original_forward


def plot_attention_on_axis(attention_weights, head_idx, ax, tokens=None, title=None, lock_color_range=True, show_colorbar=True, show_labels=True):
    """
    Plot attention heatmap for a specific head on a provided matplotlib axis
    
    Args:
        attention_weights: Tensor of shape (B, nh, T, T) with attention weights
        head_idx: Which head to visualize (0-indexed)
        ax: Matplotlib axis to plot on
        tokens: Optional list of token strings for axis labels
        title: Optional title for the plot
        lock_color_range: If True, lock color scale to [0.0, 1.0]. If False, auto-scale.
        show_colorbar: Whether to show colorbar
        show_labels: Whether to show token labels (if tokens provided)
    """
    # Extract attention weights for the specified head
    # Shape: (B, nh, T, T) -> (T, T) for the first batch and specified head
    att_matrix = attention_weights[0, head_idx].cpu().numpy()
    
    # Determine if we should show token labels
    display_labels = show_labels and tokens is not None and len(tokens) <= 15
    
    # Set color range parameters
    heatmap_kwargs = {
        'cmap': 'Blues',
        'ax': ax,
        'cbar': show_colorbar,
        'xticklabels': tokens if display_labels else False,
        'yticklabels': tokens if display_labels else False,
        'annot': False,
        'fmt': '.3f'
    }
    
    if lock_color_range:
        heatmap_kwargs['vmin'] = 0.0
        heatmap_kwargs['vmax'] = 1.0
    
    # Create heatmap
    sns.heatmap(att_matrix, **heatmap_kwargs)
    
    if title:
        ax.set_title(title)
    
    # Rotate labels if showing tokens
    if display_labels:
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)


def plot_attention_heatmap(attention_weights, head_idx, tokens=None, title=None, figsize=(12, 10), lock_color_range=True):
    """
    Plot attention heatmap for a specific head
    
    Args:
        attention_weights: Tensor of shape (B, nh, T, T) with attention weights
        head_idx: Which head to visualize (0-indexed)
        tokens: Optional list of token strings for axis labels
        title: Optional title for the plot
        figsize: Figure size tuple
        lock_color_range: If True, lock color scale to [0.0, 1.0]. If False, auto-scale.
    """
    # Extract attention weights for the specified head
    # Shape: (B, nh, T, T) -> (T, T) for the first batch and specified head
    att_matrix = attention_weights[0, head_idx].cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Always show tokens if provided, with better formatting
    show_labels = tokens is not None and len(tokens) <= 50  # Only show labels if reasonable number
    
    # Set color range parameters
    heatmap_kwargs = {
        'cmap': 'Blues',
        'xticklabels': tokens if show_labels else True,
        'yticklabels': tokens if show_labels else True,
        'cbar_kws': {'label': 'Attention Weight'},
        'annot': False,  # Don't annotate with values to keep clean
        'fmt': '.3f'
    }
    
    if lock_color_range:
        heatmap_kwargs['vmin'] = 0.0
        heatmap_kwargs['vmax'] = 1.0
    
    # Create heatmap
    sns.heatmap(att_matrix, **heatmap_kwargs)
    
    plt.title(title or f'Attention Heatmap - Head {head_idx}')
    plt.xlabel('Key Position (tokens being attended to)')
    plt.ylabel('Query Position (tokens doing the attending)')
    
    # Rotate x-axis labels if tokens are provided for better readability
    if show_labels:
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_multi_head_attention(attention_weights, tokens=None, layer_idx=None, max_heads=8, lock_color_range=True):
    """
    Plot attention heatmaps for multiple heads in a grid
    
    Args:
        attention_weights: Tensor of shape (B, nh, T, T) with attention weights
        tokens: Optional list of token strings for axis labels
        layer_idx: Layer index for title
        max_heads: Maximum number of heads to display
        lock_color_range: If True, lock color scale to [0.0, 1.0]. If False, auto-scale.
    """
    B, nh, T, T_key = attention_weights.shape
    num_heads = min(nh, max_heads)
    
    # Calculate grid dimensions
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Set color range parameters
    heatmap_kwargs = {
        'cmap': 'Blues',
        'cbar': True,
        'xticklabels': False,
        'yticklabels': False
    }
    
    if lock_color_range:
        heatmap_kwargs['vmin'] = 0.0
        heatmap_kwargs['vmax'] = 1.0
    
    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes[0]
        
        # Extract attention weights for this head
        att_matrix = attention_weights[0, head_idx].cpu().numpy()
        
        # Create heatmap
        sns.heatmap(att_matrix, ax=ax, **heatmap_kwargs)
        
        ax.set_title(f'Head {head_idx}')
    
    # Hide extra subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    
    layer_title = f'Layer {layer_idx} - ' if layer_idx is not None else ''
    fig.suptitle(f'{layer_title}Attention Patterns Across Heads', fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_attention_patterns(attention_weights, tokens=None, head_idx=0):
    """
    Analyze and print statistics about attention patterns
    
    Args:
        attention_weights: Tensor of shape (B, nh, T, T) with attention weights
        tokens: Optional list of token strings
        head_idx: Which head to analyze
    """
    att_matrix = attention_weights[0, head_idx].cpu().numpy()
    T = att_matrix.shape[0]
    
    print(f"Attention Analysis for Head {head_idx}")
    print(f"Sequence length: {T}")
    print(f"Attention matrix shape: {att_matrix.shape}")
    print()
    
    # Analyze attention distribution
    print("Attention Statistics:")
    print(f"Mean attention weight: {att_matrix.mean():.4f}")
    print(f"Max attention weight: {att_matrix.max():.4f}")
    print(f"Min attention weight: {att_matrix.min():.4f}")
    print(f"Std attention weight: {att_matrix.std():.4f}")
    print()
    
    # Find positions with highest attention
    print("Top 5 attention positions (query -> key):")
    flat_indices = np.argsort(att_matrix.flatten())[-5:]
    for idx in reversed(flat_indices):
        row, col = np.unravel_index(idx, att_matrix.shape)
        weight = att_matrix[row, col]
        query_token = tokens[row] if tokens else f"pos_{row}"
        key_token = tokens[col] if tokens else f"pos_{col}"
        print(f"  {query_token} -> {key_token}: {weight:.4f}")
    print()
    
    # Analyze attention to self vs others
    self_attention = np.diag(att_matrix).mean()
    other_attention = (att_matrix.sum() - np.diag(att_matrix).sum()) / (T * (T - 1))
    print(f"Average self-attention: {self_attention:.4f}")
    print(f"Average attention to others: {other_attention:.4f}") 