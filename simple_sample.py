"""
Simple script to sample stories from a trained model with temperature control, stopping at EOS tokens
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from tqdm import tqdm

# Configuration
out_dir = 'out/dataset_tests/camstories_10k_base_rope_run2'
num_samples = 200
max_new_tokens = 500
temperature = 0.7
first_token_temperature = 1.2  # Higher temperature until first non-EOS token
top_k = 200
seed = 1337
save_to_file = True  # Save stories to text file
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

print(f"Using device: {device}")
torch.manual_seed(seed)
if 'cuda' in device:
    torch.cuda.manual_seed(seed)

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# Clean up state dict
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Load the meta.pkl to get stoi/itos
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    itos = meta['itos']
    print(f"Vocabulary size: {len(stoi)}")
    
    # Find <|endoftext|> token
    eos_token_id = None
    for token, idx in stoi.items():
        if token == '<|endoftext|>':
            eos_token_id = idx
            print(f"Found <|endoftext|> token with ID: {eos_token_id}")
            break
    
    if eos_token_id is None:
        # Try common EOS token IDs and look for endoftext variants
        for candidate_id in [0, 1, 2, 3]:
            if candidate_id < len(itos):
                token = itos[candidate_id]
                if 'endoftext' in token.lower():
                    eos_token_id = candidate_id
                    print(f"Found endoftext token: '{token}' with ID: {eos_token_id}")
                    break
    
    # Improved decode function with proper spacing, ## removal, punctuation handling, and EOS token filtering
    def decode(token_ids):
        tokens = [itos[idx] for idx in token_ids if idx < len(itos)]
        text = ""
        punctuation = {'.', ',', '!', '?', ':', ';'}
        quotes = {"'", '"'}
        
        for i, token in enumerate(tokens):
            if token == '##':
                continue  # Skip ## tokens entirely
            elif token == '<|endoftext|>':
                continue  # Skip EOS tokens - don't render them
            elif token.startswith('##'):
                # Remove ## prefix and don't add space
                text += token[2:]
            elif token in punctuation:
                # Punctuation: no space before, add space after (unless it's the last token)
                text += token
                if i < len(tokens) - 1 and not tokens[i+1].startswith('<|'):
                    text += ' '
            elif token in quotes:
                # Quotes: no space before, no automatic space after
                text += token
            elif i > 0 and not tokens[i-1].endswith('##') and not token.startswith('<|') and not token.endswith('|>') and tokens[i-1] not in (punctuation | quotes):
                # Add space before token unless:
                # - previous token ended with ##
                # - this is a special token
                # - previous token was punctuation or quotes
                text += ' ' + token
            else:
                text += token
        return text
    
else:
    print("No meta.pkl found - cannot proceed without vocabulary mapping")
    exit(1)

def generate_with_eos_stop(model, start_ids, max_new_tokens, temperature=1.0, first_token_temperature=None, top_k=None, eos_token_id=None):
    """Generate tokens, stopping when EOS token is encountered (but not if first 2 tokens are both EOS)
    
    Args:
        first_token_temperature: Higher temperature to use until first non-EOS token is generated
    """
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(start_ids, list):
        idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        idx = start_ids
    
    original_length = idx.size(1)
    tokens_generated = 0
    found_non_eos = False
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop if sequence is too long
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]  # Get last token logits
            
            # Choose temperature: use first_token_temperature until we get a non-EOS token
            current_temp = temperature
            if first_token_temperature is not None and not found_non_eos:
                current_temp = first_token_temperature
            
            if current_temp == 0.0:
                # Greedy decoding
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Sample with temperature
                logits = logits / current_temp
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            tokens_generated += 1
            
            # Track if we've found a non-EOS token
            if eos_token_id is not None and idx_next.item() != eos_token_id:
                found_non_eos = True
            
            # Stop if we hit EOS token, but only if we're past the first 2 generated tokens
            # or if the first 2 tokens aren't both EOS
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                if tokens_generated > 2:
                    break
                elif tokens_generated == 2:
                    # Check if first two generated tokens are both EOS
                    generated_tokens = idx[0, original_length:].tolist()
                    if len(generated_tokens) >= 2 and generated_tokens[0] == eos_token_id and generated_tokens[1] == eos_token_id:
                        # First two are both EOS, continue generating
                        continue
                    else:
                        # Not both EOS, safe to stop
                        break
                elif tokens_generated == 1:
                    # First token is EOS, continue to see if second is also EOS
                    continue
    
    return idx

# Start generation from EOS token
if eos_token_id is not None:
    start_token = eos_token_id
else:
    start_token = 0  # Fallback to token 0

print(f"\nGenerating {num_samples} stories with temperature {temperature}")
print(f"First token temperature: {first_token_temperature} (until first non-EOS token)")
print(f"Starting from token: {itos[start_token]} (ID: {start_token})")
print(f"Will stop at <|endoftext|> token (ID: {eos_token_id})")
if save_to_file:
    output_file = os.path.join(out_dir, f"generated_stories_temp{temperature}_firsttemp{first_token_temperature}.txt")
    print(f"Will save stories to: {output_file}")
print("=" * 80)

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Store stories if saving to file
stories = []

with ctx:
    # Use tqdm for progress bar
    for i in tqdm(range(num_samples), desc="Generating stories", unit="story"):
        generated = generate_with_eos_stop(
            model, 
            [start_token], 
            max_new_tokens, 
            temperature=temperature,
            first_token_temperature=first_token_temperature,
            top_k=top_k, 
            eos_token_id=eos_token_id
        )
        
        story = decode(generated[0].tolist()).strip()
        
        # Print first few stories for preview
        if i < 3:
            print(f"\n--- Story {i+1} ---")
            print(story)
            print('-' * 50)
        
        # Store story for saving
        if save_to_file:
            stories.append(story)

# Save stories to file if requested
if save_to_file and stories:
    with open(output_file, 'w', encoding='utf-8') as f:
        for story in stories:
            f.write(story + '\n')
    print(f"\n✓ Saved {len(stories)} stories to {output_file}")

print(f"\nDone! Generated {num_samples} stories.")