# train a Tinystories-8M model (≈8.9M params) for quick iteration and decent output quality

out_dir = 'out-tinystories-8M'
eval_interval = 250        # frequent eval to monitor overfitting
eval_iters = 200
log_interval = 10          # print training loss every 10 iters

always_save_checkpoint = False  # only save on validation improvement
wandb_log = False                # set True via CLI if needed
wandb_project = 'tinystories'
wandb_run_name = 'tiny-8M'

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256         # context length of 256 tokens

# model: hidden_size=256, 8 layers, 16 heads → ≈8.9M parameters
n_layer = 8
n_head = 16
n_embd = 256
dropout = 0.1

learning_rate = 5e-4     # a bit lower for stable training
max_iters = 5000
lr_decay_iters = 5000    # linear decay over full training
min_lr = 5e-5            # 1/10 of initial LR
beta2 = 0.99             # smoother second moment

warmup_iters = 100       # short warmup period

# device and compilation settings (for MacBooks etc.)
# device = 'cpu'
# compile = False
