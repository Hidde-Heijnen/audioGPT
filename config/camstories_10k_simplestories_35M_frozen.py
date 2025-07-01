# train a 35M parameter model for preprocessed stories with SimpleStories tokenization
# FROZEN TOKEN & POSITION EMBEDDINGS

out_dir = 'out/camstories_10k_simplestories/35M_frozen'
eval_interval = 100        # frequent eval to monitor overfitting
eval_iters = 100
log_interval = 10          # print training loss every 10 iters

always_save_checkpoint = False  # only save on validation improvement
wandb_log = False                # set True via CLI if needed
wandb_project = 'audiogpt'

# --- IMPORTANT: freeze the token and positional embeddings ---
freeze_embeddings = True

dataset = 'camstories/10000_SimpleStories_SimpleStories-35M'
gradient_accumulation_steps = 1  # Default value
batch_size = 64

# --- Positional Encoding settings ---
# Options: "learned", "zeropad", "none"
posenc_type = "learned"

# -- Only used by "zeropad" posenc_type --
# For "learned" or "none", n_embd below will be the token embedding dimension.
# For "zeropad", the total hidden width (n_embd) is embed_dim_token + extra_dim
embed_dim_token = 512
extra_dim = 768

# model: 35M parameter model - 12 layers, 8 heads, 512 embedding (matching SimpleStories-35M)
n_layer = 12  # Increased from 8 to match SimpleStories-35M
n_head = 8
n_embd = 512  # Base embedding dimension
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 15000    # Extended training - increased from 10000
decay_lr = True
min_lr = 6e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
block_size = 512

warmup_iters = 1000
lr_decay_iters = 15000  # Match max_iters - increased from 10000

# device and compilation settings
# device = 'cuda'
# compile = False
seed = 1337 