"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
early_eval_interval = 100  # evaluation interval for first 1000 iterations
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'audiogpt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
# Positional-encoding defaults
posenc_type = 'learned'  # 'learned' | 'zeropad' | 'sinusoidal' | 'none'
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# Positional-encoding scale (multiplies positional embeddings when posenc_type is learned or sinusoidal)
posenc_scale = 1.0
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
freeze_embeddings = False # whether to freeze embedding layers
locked_embeddings = None # column name in parquet file to use for locked embeddings in the main transformer (non-shadow part) (e.g. "4096_vec"), or None to disable
shadow_audio_col = None  # None to disable, or string column name for locked audio embeddings
transformer_type = 'normal_transformer'  # 'normal_transformer' or 'shadow_audio'
audio_dim = 0 # audio embedding dimension for shadow audio transformers (automatically detected if 0 and shadow_audio_col is not None )
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# audio alignment loss
audio_alignment_loss = False
audio_alignment_lambda = 1.0
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run? (multiple GPUs)
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    # Update config for wandb logging
    config['gradient_accumulation_steps'] = gradient_accumulation_steps
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# check for locked embeddings early to determine embedding dimension
parquet_path = 'data/audio_embedding/tokens_audio_10k.parquet'
df = None  # Initialize dataframe variable

# Load parquet file once if either locked_embeddings or shadow_audio_col is specified
if locked_embeddings is not None or shadow_audio_col is not None:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    print(f"Loading parquet file from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

if locked_embeddings is not None:
    if locked_embeddings not in df.columns:
        raise ValueError(f"Column '{locked_embeddings}' not found in parquet file. Available columns: {list(df.columns)}")
    
    print(f"Detecting embedding dimension from column '{locked_embeddings}'...")
    # Get embedding dimension from first embedding
    sample_embedding = df[locked_embeddings].iloc[0]
    detected_embed_dim = len(sample_embedding)
    
    print(f"Detected embedding dimension: {detected_embed_dim}")
    print(f"Overriding n_embd from {n_embd} to {detected_embed_dim}")
    
    # Override global variables
    n_embd = detected_embed_dim
    embed_dim_token = detected_embed_dim
    
    # Update the config dict for wandb logging
    globals()['n_embd'] = n_embd
    globals()['embed_dim_token'] = embed_dim_token
    
    # Update the config dict so wandb shows correct values
    config['n_embd'] = n_embd
    config['embed_dim_token'] = embed_dim_token

# Initialize audio dimension
if shadow_audio_col is not None and transformer_type == 'shadow_audio':
    if shadow_audio_col not in df.columns:
        raise ValueError(f"Column '{shadow_audio_col}' not found in parquet file. Available columns: {list(df.columns)}")
    
    print(f"Detecting audio embedding dimension from column '{shadow_audio_col}'...")
    # Sample two embeddings to check consistency
    sample1 = df[shadow_audio_col].iloc[0]
    sample2 = df[shadow_audio_col].iloc[1]
    dim1 = len(sample1)
    dim2 = len(sample2)
    if dim1 != dim2:
        raise ValueError("Inconsistent embedding dimensions in shadow_audio_col")
    
    detected_audio_dim = dim1
    print(f"Detected audio_dim={detected_audio_dim} from column '{shadow_audio_col}'")
    config['audio_dim'] = detected_audio_dim
elif shadow_audio_col is None and transformer_type == 'shadow_audio':
    raise ValueError("shadow_audio_col required for shadow_audio transformer_type")
else:
    detected_audio_dim = 0
    config['audio_dim'] = detected_audio_dim

# model init
# Collect model args, now including optional positional encoding knobs
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# Optional positional-encoding overrides (back-compatible)
for _k in ("posenc_type", "embed_dim_token", "extra_dim", "posenc_scale"):
    if _k in globals():
        model_args[_k] = globals()[_k]

# Audio options
model_args['audio_dim'] = detected_audio_dim
model_args['transformer_type'] = transformer_type
model_args['audio_alignment_loss'] = audio_alignment_loss
model_args['audio_alignment_lambda'] = audio_alignment_lambda

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'posenc_type', 'embed_dim_token', 'extra_dim', 'posenc_scale', 'audio_dim', 'transformer_type', 'audio_alignment_loss', 'audio_alignment_lambda']:
        model_args[k] = checkpoint_model_args[k]
        # Update config for wandb logging if key exists in config
        if k in config:
            config[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'posenc_type', 'embed_dim_token', 'extra_dim', 'posenc_scale', 'audio_dim', 'transformer_type', 'audio_alignment_loss', 'audio_alignment_lambda']:
        model_args[k] = getattr(model.config, k)
        # Update config for wandb logging if key exists in config
        if k in config:
            config[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
    # Update config for wandb logging
    config['block_size'] = block_size

# load locked embeddings if specified
def load_locked_embeddings(parquet_path, embed_col, meta_path, df=None):
    """
    Load pre-existing embeddings from parquet file and match them to the vocabulary.
    Returns a tensor of shape (vocab_size, embed_dim) with embeddings.
    
    Args:
        parquet_path: Path to the parquet file (used for logging or fallback reading)
        embed_col: Column name containing embeddings
        meta_path: Path to the meta.pkl file with vocabulary info
        df: Optional pre-loaded dataframe. If None, will read parquet_path.
    """
    print(f"Loading locked embeddings from {parquet_path}, column: {embed_col}")
    
    # Load parquet file if not provided
    if df is None:
        df = pd.read_parquet(parquet_path)
    
    if 'token' not in df.columns or embed_col not in df.columns:
        raise ValueError(f"Parquet file must contain 'token' and '{embed_col}' columns")
    
    # Load meta file to get tokenizer info
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Get vocab mappings
    stoi = meta['stoi']  # string to integer mapping
    itos = meta['itos']  # integer to string mapping
    vocab_size = len(stoi)
    
    # Create token to embedding mapping from parquet
    token_to_embed = {}
    for _, row in df.iterrows():
        token = row['token']
        embed = np.array(row[embed_col])
        token_to_embed[token] = embed
    
    # Get embedding dimension from first embedding
    embed_dim = len(next(iter(token_to_embed.values())))
    print(f"Embedding dimension: {embed_dim}")
    
    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    missing_tokens = []
    
    # Fill embedding matrix
    for token, token_id in stoi.items():
        if token in token_to_embed:
            embedding_matrix[token_id] = token_to_embed[token]
        else:
            missing_tokens.append(token)
    
    if missing_tokens:
        print(f"WARNING: {len(missing_tokens)} tokens from vocabulary not found in parquet file")
        print(f"First few missing tokens: {missing_tokens[:10]}")
        # Initialize missing tokens with Gaussian noise when using locked embeddings
        for token in missing_tokens:
            token_id = stoi[token]
            # embedding_matrix[token_id] = np.zeros(embed_dim, dtype=np.float32)
            embedding_matrix[token_id] = np.random.normal(0, 0.02, embed_dim)
            print(f"Initialized token {token} with Gaussian noise")
        print(f"Initialized {len(missing_tokens)} missing tokens with Gaussian noise")
    
    print(f"Successfully loaded embeddings for {len(token_to_embed)} tokens")
    return torch.from_numpy(embedding_matrix)

if locked_embeddings is not None:
    
    if meta_path is None or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}. Cannot load locked embeddings without vocabulary mapping.")
    
    # Load the locked embeddings
    embedding_matrix = load_locked_embeddings(parquet_path, locked_embeddings, meta_path, df)
    
    # Sanity check - dimensions should match since we set them earlier
    expected_embed_dim = model.config.embed_dim_token
    if embedding_matrix.shape[1] != expected_embed_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_embed_dim}, got {embedding_matrix.shape[1]}")
    
    if embedding_matrix.shape[0] != model.config.vocab_size:
        raise ValueError(f"Vocab size mismatch: expected {model.config.vocab_size}, got {embedding_matrix.shape[0]}")
    
    # Convert to model dtype and load into model
    embedding_matrix = embedding_matrix.to(ptdtype)
    with torch.no_grad():
        model.transformer.wte.weight.copy_(embedding_matrix)
    
    print(f"Loaded locked embeddings with shape {embedding_matrix.shape}")
    
    # Automatically freeze embeddings when using locked embeddings
    freeze_embeddings = True
    print("Automatically enabling freeze_embeddings when using locked_embeddings")
    # Update config for wandb logging
    config['freeze_embeddings'] = freeze_embeddings

if shadow_audio_col is not None:
    if meta_path is None or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}. Cannot load locked audio embeddings without vocabulary mapping.")
    
    # Load the locked audio embeddings
    embedding_matrix = load_locked_embeddings(parquet_path, shadow_audio_col, meta_path, df)
    
    # Sanity check
    if embedding_matrix.shape[1] != detected_audio_dim:
        raise ValueError(f"Audio embedding dimension mismatch: expected {detected_audio_dim}, got {embedding_matrix.shape[1]}")
    
    if embedding_matrix.shape[0] != model.config.vocab_size:
        raise ValueError(f"Vocab size mismatch: expected {model.config.vocab_size}, got {embedding_matrix.shape[0]}")
    
    # Convert to model dtype and load into model
    embedding_matrix = embedding_matrix.to(ptdtype)
    with torch.no_grad():
        model.transformer.w_audio.weight.copy_(embedding_matrix)
    
    print(f"Loaded locked audio embeddings with shape {embedding_matrix.shape}")
    
    # Note: Shadow audio embeddings are always frozen by design separately, don't override freeze_embeddings (see w_audio below)
    print("Shadow audio embeddings will be frozen (they are always frozen by design)")



model.to(device)

# freeze token embedding layers if requested
if freeze_embeddings:
    print("freezing token embeddings")
    # explicitly freeze token embeddings and (by weight tying) the lm_head
    model.transformer.wte.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False  # same object as wte.weight, but keep explicit for clarity

    # --- sanity checks ---
    assert model.lm_head.weight is model.transformer.wte.weight, "lm_head weight and token embedding weight are not the same object!"
    assert not model.transformer.wte.weight.requires_grad, "Token embeddings should be frozen when freeze_embeddings=True"
    assert not model.lm_head.weight.requires_grad, "Output embeddings should be frozen when freeze_embeddings=True"

# freeze shadow audio embeddings if present (they are always frozen by design)
# w_audio is the shadow audio embedding layer (nn.Embedding) that maps token IDs to audio embeddings
if hasattr(model.transformer, 'w_audio'):
    print("freezing shadow audio embeddings (always frozen by design)")
    model.transformer.w_audio.weight.requires_grad = False
    assert not model.transformer.w_audio.weight.requires_grad, "Shadow audio embeddings should always be frozen"

# handle positional embeddings separately based on posenc_type and posenc_scale
if posenc_type == "learned" and posenc_scale != 0:
    print("positional embeddings are trainable (learned posenc_type with non-zero scale)")
    model.transformer.wpe.weight.requires_grad = True
else:
    print(f"freezing positional embeddings (posenc_type={posenc_type}, posenc_scale={posenc_scale})")
    model.transformer.wpe.weight.requires_grad = False

# --- sanity check for positional embeddings ---
if posenc_type == "learned" and posenc_scale != 0:
    assert model.transformer.wpe.weight.requires_grad, "Positional embeddings should be trainable when posenc_type=learned and posenc_scale!=0"
else:
    assert not model.transformer.wpe.weight.requires_grad, f"Positional embeddings should be frozen when posenc_type={posenc_type} or posenc_scale={posenc_scale}"

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda',enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# get parameter count for logging
raw_model = model.module if ddp else model # unwrap DDP container if needed
num_params = raw_model.get_num_params()
num_params_millions = num_params / 1e6
print(f"model has {num_params_millions:.2f}M parameters")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    # add parameter count to config for wandb logging
    config_with_params = config.copy()
    config_with_params['num_params'] = num_params
    config_with_params['num_params_millions'] = num_params_millions
    wandb.init(project=wandb_project, name=wandb_run_name, config=config_with_params)
    # log parameter count as a summary metric
    wandb.log({"model/num_params": num_params, "model/num_params_millions": num_params_millions})

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# raw_model already defined above for parameter counting
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    # use early_eval_interval for first 1000 iterations, then switch to regular eval_interval
    current_eval_interval = early_eval_interval if iter_num < 1000 else eval_interval
    if iter_num % current_eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        if torch.isnan(loss).any() or torch.isinf(loss).any():  # Lightweight check, gated if needed
            if iter_num > 1000:  # Only check after warmup to focus on your delayed NaN
                raise ValueError(f"NaN/Inf detected in loss at iter {iter_num} before backward")
        scaler.scale(loss).backward()
        # clip the gradient
        grad_norm = None  # Initialize for logging
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
                norm_type=2.0,
                error_if_nonfinite=True  # Raises RuntimeError on NaN/Inf grads
            )
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        grad_str = f", grad norm {grad_norm:.4f}" if grad_norm is not None else ""
        print(f"iter {iter_num}: loss {lossf:.4f}{grad_str}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": lossf,
                "mfu": running_mfu*100,
            }
            if grad_norm is not None:
                grad_norm_item = grad_norm.item()
                log_dict["train/grad_norm"] = grad_norm_item
            wandb.log(log_dict)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()