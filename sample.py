"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tokenizer import get_tokenizer

# Conditional import for Tokenizer
try:
    from data.tinystories.tokenizer import Tokenizer as TinyStoriesTokenizer
    TINYSTORIES_TOKENIZER_AVAILABLE = True
except ImportError:
    TINYSTORIES_TOKENIZER_AVAILABLE = False
    TinyStoriesTokenizer = None # Placeholder if import fails

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "<|endoftext|>" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# --- Tokenizer setup ---
encode = None
decode = None
dataset_name = None

if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    dataset_name = checkpoint['config']['dataset']

def parse_dataset_name(dataset_name):
    """
    Parse dataset name to extract vocab size and tokenizer type
    Examples:
    - camstories/5000 -> base_name=camstories, vocab_size=5000, tokenizer_type=None
    - camstories/5000_pmod -> base_name=camstories, vocab_size=5000, tokenizer_type=pmod
    """
    if '/' in dataset_name:
        base_name, suffix = dataset_name.split('/', 1)
    else:
        base_name = dataset_name
        suffix = ""
    
    parts = suffix.split('_')
    vocab_size = 0
    tokenizer_type = None
    
    for i, part in enumerate(parts):
        try:
            vocab_size = int(part)
            # Check if there's a tokenizer type after the vocab size
            if i + 1 < len(parts):
                tokenizer_type = parts[i + 1]
            break
        except ValueError:
            continue
    
    return base_name, vocab_size, tokenizer_type

def get_tokenizer_name(tokenizer_type):
    """
    Map tokenizer type to tokenizer name for get_tokenizer function
    """
    if tokenizer_type == 'pmod':
        return 'word_level_pmod'
    else:
        return 'word_level'  # Default to word_level

if dataset_name == 'tinystories':
    if not TINYSTORIES_TOKENIZER_AVAILABLE:
        # Provide a more informative error or fallback if TinyStoriesTokenizer is None
        error_msg = "TinyStories dataset specified, but 'data.tinystories.tokenizer.Tokenizer' could not be imported. "
        error_msg += "Ensure tokenizer.py is in data/tinystories/ and the environment is set up correctly."
        raise ImportError(error_msg)
    
    print("Using TinyStories custom tokenizer.")

    # Define LocalTokenizerConfig, similar to what prepare.py might use internally
    class LocalTokenizerConfig:
        def __init__(self, name):
            self.name = name

    tokenizer_config_name = "EleutherAI/gpt-neo-125M" # Default from tinystories/prepare.py
    top_k_val = 10000 # Default from tinystories/prepare.py
    
    # Construct path to token_counts.json
    # Assumes sample.py is run from the project root directory (e.g., audioGPT/)
    # and data/tinystories/ is a subdirectory.
    token_counts_path = os.path.join('data', dataset_name, 'tinystories_token_counts.json')

    if not os.path.exists(token_counts_path):
        raise FileNotFoundError(
            f"Token counts file not found for TinyStories: {token_counts_path}. "
            f"Please ensure 'prepare.py' for the 'tinystories' dataset has been run successfully."
        )

    custom_tokenizer = TinyStoriesTokenizer(
        config=LocalTokenizerConfig(tokenizer_config_name),
        k=top_k_val,
        file_path=token_counts_path,
        device=device # Use the main script's device (e.g., 'cuda:0' or 'cpu')
    )

    # Encoder: string -> tensor of token IDs (shape (1, seq_len))
    # The custom_tokenizer.encoder handles add_special_tokens and returns a tensor on the correct device.
    encode = lambda s: custom_tokenizer.encoder(s, add_special_tokens=True)

    # Decoder: list of token IDs (for one sample) -> string
    # custom_tokenizer.decoder expects a 2D list/tensor of tokens.
    decode = lambda l: custom_tokenizer.decoder(torch.tensor([l], dtype=torch.long, device=device))[0]

elif dataset_name and dataset_name.startswith('camstories'):
    # Handle camstories datasets using the tokenizer.py get_tokenizer function
    print(f"Using camstories tokenizer for dataset: {dataset_name}")
    
    base_name, vocab_size, tokenizer_type = parse_dataset_name(dataset_name)
    tokenizer_name = get_tokenizer_name(tokenizer_type)
    
    print(f"Base name: {base_name}, Vocab size: {vocab_size}, Tokenizer type: {tokenizer_type}, Tokenizer name: {tokenizer_name}")
    
    # Get the tokenizer using the same method as prepare.py
    tokenizer = get_tokenizer(
        tokenizer_name=tokenizer_name,
        dataset_name=base_name,
        vocab_size=vocab_size,
        built_vocab=True
    )
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Set up encode/decode functions
    encode = lambda s: tokenizer(s, return_tensors='pt', add_special_tokens=True)['input_ids'][0].tolist()
    
    def custom_decode(token_ids):
        """Custom decode function that handles punctuation spacing correctly"""
        # Convert token IDs back to tokens
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        
        # Join tokens with proper spacing
        result = ""
        for i, token in enumerate(tokens):
            if i == 0:
                result += token
            elif token.startswith(('!', '?', '.', ',', ';', ':', "'", '"', ')', ']', '}')) or token in ['!', '?', '.', ',', ';', ':', "'", '"', ')', ']', '}']:
                # Don't add space before punctuation
                result += token
            elif result.endswith(('(', '[', '{')):
                # Don't add space after opening brackets
                result += token
            else:
                # Add space before regular tokens
                result += " " + token
        
        return result
    
    decode = custom_decode

else:
    # Original logic for meta.pkl or tiktoken
    load_meta = False
    print(f"dataset_name: {dataset_name}")
    # Check for meta.pkl if not using tinystories custom tokenizer path
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        # dataset_name would have been set earlier
        meta_path = os.path.join('data', dataset_name, 'meta.pkl')
        if os.path.exists(meta_path):
            load_meta = True
        else:
            print(f"Meta file not found at {meta_path} for dataset '{dataset_name}'.")

    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        # This assumes meta['stoi'] and meta['itos'] are for character-level or simple tokenization
        # If meta.pkl was from tinystories but this path was taken, this might be an issue.
        # The primary `if dataset_name == 'tinystories'` block should handle it.
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s] # Assumes s is iterable and c is in stoi (char-level)
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # Fallback to GPT-2 encodings if not tinystories and no meta.pkl
        print("No meta.pkl found or not using a recognized custom tokenizer setup, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

if dataset_name == 'tinystories' and TINYSTORIES_TOKENIZER_AVAILABLE:
    # encode() from custom_tokenizer returns a tensor (usually shape (1, T))
    x = encode(start)
    # Ensure x is on the correct device, though custom_tokenizer.encoder should handle it.
    x = x.to(device)
    if x.dim() == 1: # Should already be 2D from tokenizer, but as a safeguard
        x = x.unsqueeze(0)
else:
    # Original logic: encode returns a list of ints
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
