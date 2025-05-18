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
gradient_accumulation_steps = 16 # Target: 16
batch_size = 80            # Target: 80
block_size = 512           # Target: 512, context length

# model: hidden_size=256, 8 layers, 16 heads → ≈8.9M parameters
n_layer = 8
n_head = 16
n_embd = 256
dropout = 0.0            # Updated from 0.1 to 0.0 based on JSON (attention_dropout, etc.)

learning_rate = 5e-4     # Target: 5e-4
max_iters = 5000
# lr_decay_iters, min_lr, and warmup_iters are removed as decay_lr = False
decay_lr = False         # Target: lr_schedule = constant
min_lr = 5e-5            # This will be ignored due to decay_lr = False, kept for reference or easy toggle
beta1 = 0.9              # Target: adam_beta1=0.9 (explicitly set)
beta2 = 0.95             # Target: adam_beta2 = 0.95 (changed from 0.99)
weight_decay = 0.1       # Target: wd=0.1 (explicitly set)

warmup_iters = 100       # This will be ignored due to decay_lr = False for LR scheduling, kept for reference

# device and compilation settings
dtype = 'float32'        # Added based on JSON torch_dtype
# device = 'cpu'
# compile = False
