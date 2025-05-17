"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import json

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Load special tokens and extract characters from them
# Path to special_tokens_map.json relative to prepare.py (data/shakespeare_char/ -> data/tinystories/)
script_dir = os.path.dirname(__file__)
# Construct path to data/tinystories/special_tokens_map.json
# Goes up one level from shakespeare_char to data/, then into tinystories/
special_tokens_map_path = os.path.join(script_dir, '..', 'tinystories', 'special_tokens_map.json')

defined_special_tokens_content = {}
special_char_set = set()

if os.path.exists(special_tokens_map_path):
    with open(special_tokens_map_path, 'r') as f:
        special_tokens_data = json.load(f)
    for token_name, token_spec in special_tokens_data.items():
        if isinstance(token_spec, dict) and 'content' in token_spec:
            content = token_spec['content']
            defined_special_tokens_content[token_name] = content
            if content: # Make sure content is not None or empty
                special_char_set.update(list(content))
else:
    print(f"Warning: Special tokens map file not found at {special_tokens_map_path}")

# get all the unique characters that occur in this text
# and include characters from special tokens
chars = sorted(list(set(data).union(special_char_set)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'special_tokens': defined_special_tokens_content,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
