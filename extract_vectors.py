# Vector Extraction and Listening Tool

# Purpose: Extract and save vectors from specified points in the model during generation, including shadow audio vectors, for listening or analysis.

# %% [markdown]
## Imports and Setup
# Collect all imports here

import os
import pickle
import torch
from contextlib import nullcontext
import argparse
import ast
import numpy as np
from model import GPTConfig, GPT
from tokenizer import get_tokenizer  # Adjust if needed

# %% [markdown]
## Parameters
# Expanded extraction points based on model analysis
# Interesting points:
# - 'wte': Main token embeddings (locked if locked_embeddings set in train.py)
# - 'w_audio': Shadow audio embeddings (always locked if shadow_audio in model.py)
# - 'after_posenc': After positional encoding addition (if enabled in config)
# - Per block (layer_idx):
#   - 'after_attn': After attention (y and audio_mixed if shadow)
#   - 'after_attn_resid': After attention residual (x + y, audio + audio_mixed)
#   - 'after_mlp': After MLP
#   - 'after_mlp_resid': After MLP residual
# - 'final_ln': After final LayerNorm on token path
# - 'audio_final_ln': After final LayerNorm on audio path (if shadow)
extract_points = [
    ('wte',),
    ('w_audio',),
    ('after_posenc',),
    (0, 'after_attn', 0),  # Layer 0 after attn, head 0
    (0, 'after_attn_resid'),
    (0, 'after_mlp'),
    (0, 'after_mlp_resid'),
    ('final_ln',),
    ('audio_final_ln',)
]

init_from = 'resume'
out_dir = 'out/camstories_10k_shadow_audio_run1'
start = '<|endoftext|>'
num_samples = 1
max_new_tokens = 100
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16'
listen_index = -1  # Position in sequence to listen to (-1 for last)

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
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

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
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print(f'No meta.pkl found, using default camstories tokenizer')
        tokenizer = get_tokenizer('word_level', 'camstories', 10000, built_vocab=True)
        encode = lambda s: tokenizer(s, return_tensors='pt', add_special_tokens=True)['input_ids'][0].tolist()
        decode = lambda l: tokenizer.decode(l)
else:
    raise ValueError('No dataset found in checkpoint config')

# %% [markdown]
## Setup Hooks

captured_vectors = {}

def capture_embedding(module, input, output):
    captured_vectors[module.__class__.__name__] = output.detach().cpu()

def capture_after_posenc():
    # This would require modifying forward; instead, run a dummy forward and capture
    pass  # Placeholder; implement if needed

# Hook setup
hooks = []
for point in extract_points:
    if len(point) == 1:
        if point[0] == 'wte':
            module = model.transformer.wte
            hooks.append(module.register_forward_hook(lambda m,i,o: captured_vectors.update({'wte': o.detach().cpu()})))
        elif point[0] == 'w_audio' and hasattr(model.transformer, 'w_audio'):
            module = model.transformer.w_audio
            hooks.append(module.register_forward_hook(lambda m,i,o: captured_vectors.update({'w_audio': o.detach().cpu()})))
        elif point[0] == 'final_ln':
            module = model.transformer.ln_f
            hooks.append(module.register_forward_hook(lambda m,i,o: captured_vectors.update({'final_ln': o.detach().cpu()})))
        elif point[0] == 'audio_final_ln' and hasattr(model, 'audio_ln_f'):
            module = model.audio_ln_f
            hooks.append(module.register_forward_hook(lambda m,i,o: captured_vectors.update({'audio_final_ln': o.detach().cpu()})))
    else:
        layer_idx = point[0]
        point_name = point[1]
        head_idx = point[2] if len(point) > 2 else None
        block = model.transformer.h[layer_idx]
        if 'attn' in point_name:
            def attn_hook(m, i, o):
                key = (layer_idx, 'after_attn', head_idx)
                if isinstance(o, tuple):
                    y, audio_mixed = o
                    captured_vectors[key] = {'token': y.detach().cpu(), 'audio': audio_mixed.detach().cpu()}
                else:
                    captured_vectors[key] = o.detach().cpu()
                if head_idx is not None:
                    hs = m.n_embd // m.n_head
                    head_out = o[:, head_idx * hs : (head_idx + 1) * hs].detach().cpu() if not isinstance(o, tuple) else o[0][:, head_idx * hs : (head_idx + 1) * hs].detach().cpu()
                    captured_vectors[key + ('head',)] = head_out
            hooks.append(block.attn.register_forward_hook(attn_hook))
        if 'mlp' in point_name:
            def mlp_hook(m, i, o):
                captured_vectors[(layer_idx, 'after_mlp')] = o.detach().cpu()
            hooks.append(block.mlp.register_forward_hook(mlp_hook))
        # For residuals, would need to hook multiple and compute; simplify by capturing outputs

# %% [markdown]
## Generation and Extraction

# Encode start prompt (simplified, adjust for your tokenizer)
start_ids = encode(start)  # Assume encode function is defined
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))  # Assume decode is defined

# For residuals and after_posenc, these might require custom logic or additional hooks

# %% [markdown]
## Audio Playback
# Moved playback here, removed saving

from IPython.display import Audio, display
import numpy as np

rate = 22050
spacing_duration = 1.0
spacing = np.zeros(int(rate * spacing_duration))

def play_vector(vec, label):
    if isinstance(vec, torch.Tensor):
        vec = vec.flatten().numpy()
    elif isinstance(vec, dict):
        for k, v in vec.items():
            play_vector(v, f'{label}_{k}')
        return
    display(Audio(vec, rate=rate))
    print(label)

# Play individuals
for key, vec in captured_vectors.items():
    play_vector(vec, str(key))

# Concatenated
all_audio = []
for key in sorted(captured_vectors.keys()):
    vec = captured_vectors[key]
    if isinstance(vec, dict):
        vec = vec.get('audio', vec.get('token')).flatten().numpy()
    else:
        vec = vec.flatten().numpy()
    all_audio.append(vec)
    all_audio.append(spacing)
concatenated = np.concatenate(all_audio)
display(Audio(concatenated, rate=rate))
print('Concatenated evolution')

# %% [markdown]
## Listen to Embeddings
# Extract and play main/shadow embedding vectors (e.g., average or specific token)

token_id = 0  # Example token to 'listen' to
main_emb = model.transformer.wte.weight[token_id].detach().cpu()
play_vector(main_emb, 'Main Embedding')

if hasattr(model.transformer, 'w_audio'):
    shadow_emb = model.transformer.w_audio.weight[token_id].detach().cpu()
    play_vector(shadow_emb, 'Shadow Audio Embedding') 

# %% [markdown]
## Cleanup

for hook in hooks:
    hook.remove() 