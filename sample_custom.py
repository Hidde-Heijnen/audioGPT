"""
Custom sample script that prevents EOS tokens immediately after the prompt
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
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
eos_token_id = None

if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    dataset_name = checkpoint['config']['dataset']

def parse_dataset_name(dataset_name):
    """
    Parse dataset name to extract vocab size and tokenizer type
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

if dataset_name and dataset_name.startswith('camstories'):
    # Handle camstories datasets using the tokenizer.py get_tokenizer function
    print(f"Using camstories tokenizer for dataset: {dataset_name}")
    
    base_name, vocab_size, tokenizer_type = parse_dataset_name(dataset_name)
    
    # Check if this is a HuggingFace tokenizer format (e.g., SimpleStories_SimpleStories-35M)
    if tokenizer_type and tokenizer_type.startswith('SimpleStories'):
        # Extract the full model name from the suffix
        suffix = dataset_name.split('/', 1)[1]  # Get everything after "camstories/"
        # Find the part after the vocab size
        parts = suffix.split('_')
        model_parts = []
        found_vocab = False
        for part in parts:
            if found_vocab:
                model_parts.append(part)
            elif part.isdigit():
                found_vocab = True
        
        if model_parts:
            model_name = '/'.join(model_parts)  # Join with slash for HuggingFace format
            tokenizer_name = "huggingface"
            print(f"Detected HuggingFace tokenizer: {model_name}")
            
            # Get the tokenizer using HuggingFace format
            tokenizer = get_tokenizer(
                tokenizer_name=tokenizer_name,
                dataset_name=base_name,
                vocab_size=vocab_size,
                built_vocab=True,
                model_name=model_name
            )
        else:
            # Fallback to word-level if we can't parse the model name
            tokenizer_name = get_tokenizer_name(tokenizer_type)
            print(f"Could not parse HF model name, using word-level. Base name: {base_name}, Vocab size: {vocab_size}, Tokenizer type: {tokenizer_type}, Tokenizer name: {tokenizer_name}")
            
            tokenizer = get_tokenizer(
                tokenizer_name=tokenizer_name,
                dataset_name=base_name,
                vocab_size=vocab_size,
                built_vocab=True
            )
    else:
        # Use the standard word-level tokenizer logic
        tokenizer_name = get_tokenizer_name(tokenizer_type)
        print(f"Base name: {base_name}, Vocab size: {vocab_size}, Tokenizer type: {tokenizer_type}, Tokenizer name: {tokenizer_name}")
        
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name=base_name,
            vocab_size=vocab_size,
            built_vocab=True
        )
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Get EOS token ID for filtering - use the actual [EOS] token, not <|endoftext|>
    eos_encoding = tokenizer('[EOS]', return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
    eos_token_id = eos_encoding[0] if len(eos_encoding) == 1 else None
    print(f"EOS token ID: {eos_token_id}")
    
    # Set up encode/decode functions - disable special tokens for more natural continuation
    encode = lambda s: tokenizer(s, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
    
    def custom_decode(token_ids):
        """Custom decode function that handles punctuation spacing and subword tokens correctly"""
        # Convert token IDs back to tokens
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        
        # Join tokens with proper spacing
        result = ""
        for i, token in enumerate(tokens):
            if i == 0:
                result += token
            elif token.startswith(' ##'):
                # Handle subword tokens (e.g., " ##se ##y" -> "sey")
                # Remove the space and ## prefix
                subword = token[3:]  # Remove " ##"
                result += subword
            elif token.startswith('##'):
                # Handle subword tokens without leading space (e.g., "##se" -> "se") 
                subword = token[2:]  # Remove "##"
                result += subword
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
    # Fallback for other datasets
    print("Using fallback tokenizer setup")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Custom generation function that avoids immediate EOS
def generate_with_eos_filter(model, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
    """
    Generate tokens while filtering out EOS tokens in the first few positions
    """
    model.eval()
    original_length = idx.size(1)
    
    for i in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step
        logits = logits[:, -1, :]
        
        # Filter out EOS token for the first few generated tokens
        if eos_token_id is not None and i < 3:  # Filter EOS for first 3 generated tokens
            logits[:, eos_token_id] = float('-inf')
        
        # Handle temperature=0 case (greedy decoding)
        if temperature == 0.0:
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            # greedy decoding - select the most likely token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Optional: stop if we hit EOS after the initial filtering period
        if eos_token_id is not None and i >= 3 and idx_next.item() == eos_token_id:
            break
    
    return idx

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# Original logic: encode returns a list of ints
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(f"Input prompt: '{start}'")
print(f"Encoded as: {start_ids}")

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate_with_eos_filter(model, x, max_new_tokens, temperature=temperature, top_k=top_k, eos_token_id=eos_token_id)
            generated_text = decode(y[0].tolist())
            print(f"Sample {k+1}:")
            print(generated_text)
            print('---------------') 