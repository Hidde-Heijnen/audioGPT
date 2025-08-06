# train a 35M parameter model for preprocessed stories with SimpleStories tokenization

out_dir = 'out/camstories_10k_cased_ss_tok/35M'
eval_interval = 100        # regular eval interval after 1000 iterations
early_eval_interval = 25  # eval interval for first 1000 iterations
eval_iters = 25
log_interval = 3  

always_save_checkpoint = False  # only save on validation improvement
wandb_log = False                # set True via CLI if needed
wandb_project = 'dataset-comparison'

dataset = 'camstories/10000_cased_ss_tok'
gradient_accumulation_steps = 1  # Default value
batch_size = 512

# --- Positional Encoding settings ---
# Options: "learned", "zeropad", "none", "rope"
posenc_type = "rope"

# -- Only used by "zeropad" posenc_type --
# For "learned" or "none", n_embd below will be the token embedding dimension.
# For "zeropad", the total hidden width (n_embd) is embed_dim_token + extra_dim
embed_dim_token = 512

# model: 35M parameter model - 12 layers, 8 heads, 512 embedding (matching SimpleStories-35M)
n_layer = 12  # Increased from 8 to match SimpleStories-35M
n_head = 8
n_embd = 512  # Base embedding dimension
dropout = 0.0
bias = False

learning_rate = 4e-4
max_iters = 2500    # Extended training - increased from 10000
decay_lr = True
min_lr = 4e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
block_size = 512

warmup_iters = 125
lr_decay_iters = 2500  # Match max_iters - increased from 10000

# device and compilation settings
# device = 'cuda'
# compile = False
seed = 1337 