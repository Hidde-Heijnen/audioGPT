# train a model with locked 4096-dimensional embeddings from parquet file

out_dir = 'out/camstories_10k_locked_embeddings'
eval_interval = 500        # regular eval interval after 1000 iterations
early_eval_interval = 150  # eval interval for first 1000 iterations
eval_iters = 100
log_interval = 10          # print training loss every 10 iters

always_save_checkpoint = False  # only save on validation improvement
wandb_log = False                # set True via CLI if needed
wandb_project = 'audiogpt'

dataset = 'camstories/10000'  # use the normal camstories dataset
gradient_accumulation_steps = 1  # Default value
batch_size = 64

# --- Locked Embeddings ---
locked_embeddings = '4096_vec'  # use 4096-dimensional embeddings from parquet
freeze_embeddings = True        # will be automatically set, but explicit here

# --- Positional Encoding settings ---
# Options: "learned", "zeropad", "none"
posenc_type = "none"

# model: n_embd will be automatically set to 4096 from parquet file
n_layer = 8
n_head = 16  # 4096 must be divisible by n_head, so using 16 instead of 12
# n_embd will be automatically detected as 4096 from parquet file
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 5000    # extended training
decay_lr = True
min_lr = 6e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
block_size = 512

warmup_iters = 500
lr_decay_iters = 5000  # match max_iters

# device and compilation settings
# device = 'cuda'
# compile = False
seed = 1337 