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
# - (layer, 'after_audio_attn_resid'): after audio attn residual
# - (layer, 'after_ln2'): after ln2
# - (layer, 'after_mlp'): after mlp
# - (layer, 'after_mlp_resid'): after mlp residual
# - 'final_ln': final ln (token)
# - 'audio_final_ln': final audio ln (shadow)
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
    (0, 'after_audio_attn'),
    (1, 'after_audio_attn'),
    (2, 'after_audio_attn'),
    (3, 'after_audio_attn'),
    (4, 'after_audio_attn'),
    (5, 'after_audio_attn'),
    # (0, 'after_mlp'),
    # (0, 'after_mlp_resid'),
    # ('final_ln',),
    ('audio_final_ln',)
]

init_from = 'resume'
# out_dir = 'out/camstories_10k_shadow_audio_run1'
out_dir = "out/camstories_10k_shadow_audio_auxloss_run2" 
start = '<|endoftext|> Frankie was so tired so he went to'
num_samples = 1
max_new_tokens = 100
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu' # 'cpu' or 'cuda'
dtype = 'bfloat16'
listen_index = 8  # Position in sequence to listen to (-1 for last)

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