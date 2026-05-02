# Tiny GPT config — same architecture, much longer training
out_dir = 'out-micro'
eval_interval = 500
eval_iters = 200
log_interval = 50
always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 48

n_layer = 3
n_head = 4
n_embd = 48
dropout = 0.2
bias = False

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

device = 'mps'
compile = False