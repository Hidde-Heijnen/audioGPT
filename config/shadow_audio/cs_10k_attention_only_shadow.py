# train an attention-only shadow audio model with locked audio embeddings from parquet file

out_dir = 'out/shadow_audio/cs_10k_attention_only_shadow'
eval_interval = 500        # regular eval interval after 1000 iterations
early_eval_interval = 100  # eval interval for first 1000 iterations
eval_iters = 100
log_interval = 10          # print training loss every 10 iters

always_save_checkpoint = False  # only save on validation improvement
wandb_log = False                # set True via CLI if needed
wandb_project = 'audiogpt'

dataset = 'camstories/10000'  # use the normal camstories dataset
gradient_accumulation_steps = 5
batch_size = 64

# --- Shadow Audio Transformer ---
transformer_type = 'attention_only_shadow'  # enable attention-only shadow audio transformer
shadow_audio_col = '4096_vec'      # column name for locked audio embeddings
# audio_dim will be automatically detected from parquet file

# --- Shadow Audio Settings ---
softmax_off_by_one = True          # enable off-by-one softmax mitigation
shadow_audio_residual = "normalised_residual"  # "unnormalised_residual" or "normalised_residual"
shadow_auxiliary_loss = "target" # "none" | "expected" | "target"


# --- Locked Embeddings ---
# locked_embeddings = '4096_vec'  # use 4096-dimensional embeddings from parquet otherwise comment out. 
# freeze_embeddings = True        # will be automatically set, but explicit here

# --- Positional Encoding settings ---
# Options: "learned", "zeropad", "none"
posenc_type = "learned"

# model: n_embd will be automatically set to 4096 from parquet file
n_layer = 6
n_head = 8  # 4096 must be divisible by n_head, so using 8 instead of 12
# n_embd will be automatically detected as 4096 from parquet file
n_embd = 512
dropout = 0.0
bias = False

learning_rate = 5e-4
max_iters = 5000    # extended training
decay_lr = True
min_lr = 6e-5
beta1 = 0.90
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
# block_size = 512
block_size = 256

warmup_iters = 1000
lr_decay_iters = 5000  # match max_iters

# device and compilation settings
# device = 'cuda'
# compile = False
seed = 1337 
dtype = 'bfloat16' 