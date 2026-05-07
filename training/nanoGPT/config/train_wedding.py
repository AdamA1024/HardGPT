# Wedding-quote model training config — sized for a single A100 / 4090.
#
# Architecture goals: large enough to write coherent multi-sentence quotes,
# small enough that int8-quantized weights fit comfortably in 16 MB QSPI flash
# on the new (RP2350-class) board.
#
#   ~5.0 M parameters  (excluding embedding table)
#   ~5.5 MB int8 weight blob (incl. embeddings, per-channel scales, layernorms)
#
# Dataset: data/wedding/{train.bin, val.bin, meta.pkl} produced by
#          data/wedding/{fetch_corpus, train_tokenizer, prepare}.py
#          ~6.6 M BPE tokens at vocab_size=2048.

out_dir = "out-wedding"
eval_interval = 500
eval_iters = 100
log_interval = 25
always_save_checkpoint = False  # only checkpoint when val improves
wandb_log = False

dataset = "wedding"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 256

# Architecture
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1
bias = False

# Optim — cosine decay, AdamW betas tuned to nanoGPT defaults
learning_rate = 3e-4
max_iters = 60000
lr_decay_iters = 60000
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 1000

# nanoGPT auto-detects vocab_size from data/<dataset>/meta.pkl when present
# — our prepare.py writes vocab_size=2048 there, so we don't override here.

# Tweak these on the host you're training on:
device = "cuda"   # set to 'cpu' or 'mps' for laptop dry-runs
compile = True    # set False on first run to surface errors faster
