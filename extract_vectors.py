# Vector Extraction and Listening Tool

# Purpose: Extract and save vectors from specified points in the model during generation, including shadow audio vectors, for listening or analysis.

# %% [markdown]
## Imports and Setup
# Collect all imports here

import os
import pickle
from contextlib import nullcontext
import torch
import argparse
import ast
import numpy as np
import matplotlib.pyplot as plt
from model import GPTConfig, GPT 
from capture_manager import CaptureManager
from tokenizer import get_tokenizer

# %% [markdown]
## Parameters

# %% Parameters

# Example extract_points: flexible tuples for points
# - 'wte': main embeddings
# - 'w_audio': audio embeddings (if shadow)
# - 'after_posenc': after positional addition (if enabled)
# - (layer, 'after_ln1'): after ln1
# - (layer, 'after_audio_ln1'): after audio ln1 (shadow)
# - (layer, 'after_attn', head?): after attn (per head if specified, token path)
# - (layer, 'after_audio_attn', head?): after audio attn mix (per head before mean if specified)
# - (layer, 'after_attn_resid'): after attn residual
# The residual is computed after attention heads are combined, so specifying a head index doesn't make sense.
# - (layer, 'after_audio_attn_resid'): after audio attn residual
# - (layer, 'after_ln2'): after ln2
# - (layer, 'after_mlp'): after mlp
# - (layer, 'after_mlp_resid'): after mlp residual
# - 'final_ln': final ln (token)
# - 'audio_final_ln': final audio ln (shadow)

# Grid visualization parameters
MAX_COLUMNS = 3  # Maximum number of columns in attention visualization grids

extract_points = [
    # ('wte',),
    ('w_audio',),
    # ('after_posenc',),
    # (0, 'after_attn', 0),  # Layer 0, after attn token, head 0
    # (0, 'after_audio_attn', 0),  # Layer 0, after audio attn, head 0
    # (0, 'after_attn_resid'),
    # (1, 'after_attn_resid'),
    # (2, 'after_attn_resid'),
    # (3, 'after_attn_resid'),
    # (4, 'after_attn_resid'),
    # (5, 'after_attn_resid'),
    (0, 'after_audio_attn', 2),
    (1, 'after_audio_attn', 2),
    (2, 'after_audio_attn', 2),
    (3, 'after_audio_attn', 2),
    (4, 'after_audio_attn', 2),
    (5, 'after_audio_attn', 2),
    # (0, 'after_mlp'),
    # (0, 'after_mlp_resid'),
    # ('final_ln',),
    ('audio_final_ln',)
]

init_from = 'resume'
# out_dir = 'out/camstories_10k_shadow_audio_run1'
out_dir = "out/shadow_audio/cs_10k_shadow_offbyone_run2" 
start = 'Frankie is very tired'
num_samples = 1
max_new_tokens = 10
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu' # 'cpu' or 'cuda'
dtype = 'bfloat16'
listen_index = 3  # Position in sequence to listen to (-1 for last)

# %% [markdown]
## Paths and Config
# Model and data paths

ckpt_path = os.path.join(out_dir, 'ckpt.pt')

# %% [markdown]
## Load Model

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

use_sink_token = checkpoint['model_args'].get('use_sink_token', False)

# %% [markdown]
## Tokenizer Setup
# Dynamically load from meta.pkl if available

dataset = checkpoint.get('config', {}).get('dataset')
if dataset:
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    if os.path.exists(meta_path):
        print(f'Loading tokenizer from {meta_path}')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
        # Use proper tokenizer instead of character-level encoding
        tokenizer = get_tokenizer('word_level', 'camstories', 10000, True)
        encode = lambda s: tokenizer(s, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
        decode = lambda l: tokenizer.decode(l)
        
        if use_sink_token:
            # Load meta to get stoi
            meta_path = os.path.join('data', dataset, 'meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                stoi = meta['stoi']
                if '<|unknown|>' not in stoi:
                    raise ValueError("<|unknown|> not found for sink token")
                sink_token_id = stoi['<|unknown|>']
            else:
                raise ValueError("meta.pkl not found for camstories dataset")
    else:
        print(f'No meta.pkl found, using default camstories tokenizer')
        tokenizer = get_tokenizer('word_level', 'camstories', 10000, built_vocab=True)
        encode = lambda s: tokenizer(s, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
        decode = lambda l: tokenizer.decode(l)
else:
    raise ValueError('No dataset found in checkpoint config')

# Generation and Extraction
start_ids = encode(start)
print(f"Start string: '{start}'")
print(f"Tokenized as: {start_ids}")
print(f"Decoded tokens: {[decode([token_id]) for token_id in start_ids]}")
print(f"Listen index: {listen_index}")
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
if use_sink_token:
    sink_prefix = torch.tensor([[sink_token_id]], dtype=torch.long, device=device)
    x = torch.cat((sink_prefix, x), dim=1)

with torch.no_grad():
    with ctx:
        with CaptureManager(model, extract_points, listen_index) as manager:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        captured_vectors = manager.captures
print(decode(y[0].tolist()))

# %% Audio Playback
from IPython.display import Audio, display

rate = 8000  # From dataset prep: resampled to 8000 Hz
spacing_duration = 1.0
spacing = np.zeros(int(rate * spacing_duration))

def play_vector(vec, label):
    vec = vec.squeeze()  # Remove singleton dimensions (e.g., batch/T/nh dims that should be 1 after slicing)

    # Expected shape after squeeze: 1D tensor representing single-channel audio samples.
    # Rationale: The audio data is single-channel; if multi-dimensional, it indicates an error in vector extraction or unexpected model output. We raise an error to catch such issues rather than silently averaging.

    if vec.dim() > 1:
        raise ValueError(f"Unexpected multi-dimensional tensor after squeeze: {vec.shape}. Expected 1D for single-channel audio.")

    vec = vec.numpy()
    print(label)  # Print label before audio display
    display(Audio(vec, rate=rate))

# %% Play individuals
for key, vec in captured_vectors.items():
    play_vector(vec, key)

# %% Normalized Concatenated
# Normalize all vectors to similar amplitude before concatenating

def extract_point_to_key(extract_point):
    """Convert extract_point tuple to the key format used by CaptureManager"""
    if len(extract_point) == 1:
        return extract_point[0]
    elif len(extract_point) == 2:
        layer, stage = extract_point
        return f"layer{layer}_{stage}"
    elif len(extract_point) == 3:
        layer, stage, head = extract_point
        return f"layer{layer}_{stage}_head{head}"
    else:
        return str(extract_point)

normalized_audio = []
# Use extract_points order instead of alphabetical sorting
for extract_point in extract_points:
    key = extract_point_to_key(extract_point)
    if key in captured_vectors:
        vec = captured_vectors[key].squeeze()

        # Expected shape after squeeze: 1D tensor representing single-channel audio samples.
        # Rationale: Same as above - single-channel audio; raise error on unexpected shapes.

        if vec.dim() > 1:
            raise ValueError(f"Unexpected multi-dimensional tensor after squeeze: {vec.shape}. Expected 1D for single-channel audio.")

        vec = vec.numpy()
        
        # Normalize to [-1, 1] range for consistent volume
        if np.max(np.abs(vec)) > 0:  # Avoid division by zero
            vec = vec / np.max(np.abs(vec))
        
        normalized_audio.append(vec)
        normalized_audio.append(spacing)

normalized_concatenated = np.concatenate(normalized_audio)
print('Normalized concatenated evolution')  # Print label before audio display
display(Audio(normalized_concatenated, rate=rate))

# %% Concatenated (Original)
all_audio = []
# Use extract_points order instead of alphabetical sorting
for extract_point in extract_points:
    key = extract_point_to_key(extract_point)
    if key in captured_vectors:
        vec = captured_vectors[key].squeeze()

        # Expected shape after squeeze: 1D tensor representing single-channel audio samples.
        # Rationale: Same as above - single-channel audio; raise error on unexpected shapes.

        if vec.dim() > 1:
            raise ValueError(f"Unexpected multi-dimensional tensor after squeeze: {vec.shape}. Expected 1D for single-channel audio.")

        vec = vec.numpy()
        all_audio.append(vec)
        all_audio.append(spacing)
concatenated = np.concatenate(all_audio)
print('Concatenated evolution (original volumes)')  # Print label before audio display
display(Audio(concatenated, rate=rate))

# %% [markdown]
## Attention Visualization
# Visualize attention patterns for specific layers and heads from extract_points

# %%
from attention_viz import capture_attention_weights, plot_attention_heatmap, plot_multi_head_attention, analyze_attention_patterns, plot_attention_on_axis

# Extract attention layer/head combinations from extract_points
# Include both individual head stages and residual stages (which show mean across heads)
attention_points = []
for point in extract_points:
    if len(point) == 3 and point[1] in ['after_attn', 'after_audio_attn']:
        layer, stage, head = point
        attention_points.append((layer, stage, head))
    elif len(point) == 2 and 'attn_resid' in point[1]:
        layer, stage = point
        attention_points.append((layer, stage, 'mean'))

print(f"Will visualize attention for: {attention_points}")

# Get token strings for better visualization  
tokens = [decode([token_id]) for token_id in y[0].tolist()]
print(f"Generated sequence tokens: {tokens}")

# %% Attention Weights Table for Listen Index
# Create tables showing what tokens the listen_index token attends to

import pandas as pd

print(f"\n=== Attention Weights for Token at Position {listen_index} ===")

# Group attention points by head/stage for organized output
heads_dict = {}
for layer, stage, head in attention_points:
    key = f"{stage}_head{head}" if head != 'mean' else f"{stage}_mean"
    if key not in heads_dict:
        heads_dict[key] = []
    heads_dict[key].append((layer, stage, head))

for group_key in sorted(heads_dict.keys()):
    group_data = heads_dict[group_key]
    print(f"\n--- {group_key} ---")
    
    # Collect attention data for all layers in this group
    attention_data_by_layer = {}
    
    for layer_idx, stage, head in group_data:
        if head == 'mean':
            print(f"Capturing attention for layer {layer_idx} (mean across all heads)...")
        else:
            print(f"Capturing attention for layer {layer_idx}, head {head}...")
        
        with torch.no_grad():
            with ctx:
                with capture_attention_weights(model, layer_idx) as attention_data:
                    logits, _ = model(x)
                
                layer_attention = attention_data['weights']
        
        if layer_attention is not None:
            # Extract attention weights for the specific token at listen_index
            seq_len = layer_attention.shape[-1]
            
            # Adjust listen_index if it's negative or out of bounds
            actual_listen_index = listen_index
            if listen_index < 0:
                actual_listen_index = seq_len + listen_index
            
            if 0 <= actual_listen_index < seq_len:
                # Get attention weights from listen_index token to all other tokens
                # Shape: [batch, heads, seq_len, seq_len]
                if head == 'mean':
                    # Average across all heads for residual stages
                    token_attention = layer_attention[0, :, actual_listen_index, :seq_len].mean(dim=0)
                else:
                    # Use specific head
                    token_attention = layer_attention[0, head, actual_listen_index, :seq_len]
                attention_data_by_layer[layer_idx] = token_attention.cpu().numpy()
            else:
                print(f"  Warning: listen_index {listen_index} (adjusted: {actual_listen_index}) out of bounds for sequence length {seq_len}")
                attention_data_by_layer[layer_idx] = None
    
    # Create and display table if we have data
    if attention_data_by_layer and any(data is not None for data in attention_data_by_layer.values()):
        # Get the sequence length from the first valid layer
        seq_len = None
        for data in attention_data_by_layer.values():
            if data is not None:
                seq_len = len(data)
                break
        
        if seq_len is not None:
            # Create column headers with token strings
            display_tokens = tokens[:seq_len]
            column_headers = [f"'{token}' (pos {i})" for i, token in enumerate(display_tokens)]
            
            # Create DataFrame
            table_data = {}
            for layer_idx, stage, head in group_data:
                if attention_data_by_layer[layer_idx] is not None:
                    if head == 'mean':
                        table_data[f"Layer {layer_idx} (mean)"] = attention_data_by_layer[layer_idx]
                    else:
                        table_data[f"Layer {layer_idx}"] = attention_data_by_layer[layer_idx]
                else:
                    if head == 'mean':
                        table_data[f"Layer {layer_idx} (mean)"] = [np.nan] * seq_len
                    else:
                        table_data[f"Layer {layer_idx}"] = [np.nan] * seq_len
            
            df = pd.DataFrame(table_data, index=column_headers)
            df = df.T  # Transpose so layers are rows and tokens are columns
            
            stage_type = "mean across all heads" if any(h == 'mean' for _, _, h in group_data) else f"individual head"
            print(f"\nAttention weights from token at position {actual_listen_index} ('{tokens[actual_listen_index]}') to all tokens:")
            print(f"Values show how much attention the listened-to token pays to each position ({stage_type})")
            
            # Format the table nicely
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 15)
            
            print(df.round(4))
            
            # Also show which tokens get the most attention
            print(f"\nTop 3 attended tokens for each layer ({group_key}):")
            for layer_idx, stage, head in group_data:
                if attention_data_by_layer[layer_idx] is not None:
                    weights = attention_data_by_layer[layer_idx]
                    top_indices = np.argsort(weights)[-3:][::-1]  # Top 3 in descending order
                    top_weights = weights[top_indices]
                    
                    top_tokens = []
                    for idx, weight in zip(top_indices, top_weights):
                        top_tokens.append(f"'{tokens[idx]}' (pos {idx}): {weight:.4f}")
                    
                    layer_label = f"Layer {layer_idx} (mean)" if head == 'mean' else f"Layer {layer_idx}"
                    print(f"  {layer_label}: {', '.join(top_tokens)}")

# %% Compare Attention Across Layers (same head/stage)
if len(attention_points) > 1:
    # Group by stage and head to compare across layers
    comparison_groups = {}
    for layer, stage, head in attention_points:
        key = f"{stage}_head{head}" if head != 'mean' else f"{stage}_mean"
        if key not in comparison_groups:
            comparison_groups[key] = []
        comparison_groups[key].append((layer, stage, head))
    
    for group_key, group_data in comparison_groups.items():
        if len(group_data) > 1:
            layers = [item[0] for item in group_data]
            stage = group_data[0][1]
            head = group_data[0][2]
            
            print(f"\nComparing attention across layers {layers} for {group_key}")
            
            # Calculate grid dimensions
            num_layers = len(layers)
            cols = min(MAX_COLUMNS, num_layers)
            rows = (num_layers + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if num_layers == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, (layer_idx, stage, head) in enumerate(group_data):
                if head == 'mean':
                    print(f"Capturing attention for layer {layer_idx} (mean across all heads)...")
                else:
                    print(f"Capturing attention for layer {layer_idx}, head {head}...")
                
                with torch.no_grad():
                    with ctx:
                        with capture_attention_weights(model, layer_idx) as attention_data:
                            logits, _ = model(x)
                        
                        layer_attention = attention_data['weights']
                
                if layer_attention is not None and i < len(axes):
                    # Get tokens for this sequence length
                    seq_len = layer_attention.shape[-1]
                    display_tokens = tokens[:seq_len]
                    
                    if head == 'mean':
                        # Average across all heads for visualization
                        averaged_attention = layer_attention.mean(dim=1, keepdim=True)
                        plot_head_idx = 0  # Use index 0 for the averaged attention
                        title_suffix = " (mean)"
                    else:
                        averaged_attention = layer_attention
                        plot_head_idx = head
                        title_suffix = ""
                    
                    # Use centralized attention plotting function
                    plot_attention_on_axis(
                        averaged_attention,
                        plot_head_idx,
                        axes[i],
                        tokens=display_tokens,
                        title=f'Layer {layer_idx}{title_suffix}',
                        lock_color_range=True,
                        show_colorbar=True,
                        show_labels=True
                    )
                else:
                    if i < len(axes):
                        axes[i].text(0.5, 0.5, f'Layer {layer_idx}\n(No data)', 
                                     ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f'Layer {layer_idx} - No Data')
            
            # Hide extra subplots
            for idx in range(len(group_data), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Attention Evolution Across Layers ({group_key})', fontsize=16)
            plt.tight_layout()
            plt.show()
else:
    print("No attention points found in extract_points, or only one point to compare.")