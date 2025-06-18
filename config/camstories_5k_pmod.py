# train a medium-sized model for preprocessed stories with word-level tokenization

out_dir = 'out/camstories_5k_pmod/medium'
eval_interval = 100        # frequent eval to monitor overfitting
eval_iters = 100
log_interval = 10          # print training loss every 10 iters

always_save_checkpoint = False  # only save on validation improvement
wandb_log = False                # set True via CLI if needed
wandb_project = 'audiogpt'

dataset = 'camstories/5000_pmod'
gradient_accumulation_steps = 1  # Default value
batch_size = 64

# --- Positional Encoding settings ---
# Options: "learned", "zeropad", "none"
posenc_type = "learned"

# -- Only used by "zeropad" posenc_type --
# For "learned" or "none", n_embd below will be the token embedding dimension.
# For "zeropad", the total hidden width (n_embd) is embed_dim_token + extra_dim
embed_dim_token = 384
extra_dim = 768

# model: hidden_size=400, 6 layers, 8 heads
n_layer = 6
n_head = 8
n_embd = 384  # Base embedding dimension (doesn't get used for zeropad)
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 10000    # extended for medium training
decay_lr = True
min_lr = 6e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
block_size = 512

warmup_iters = 1000
lr_decay_iters = 10000  # match max_iters

# device and compilation settings
# device = 'cuda'
# compile = False
seed = 1337 