# Wedding-quote model training config — sized for a single A100 / 4090.
#
# Target use: SHORT romantic / wedding quotes (~10-30 words, ~15-40 BPE
# tokens). Output length is capped at inference time via max_new_tokens and
# stop-on-<eos> — block_size below is the context window, not the max output
# length.
#
# Architecture is locked to fit a single-chip RP2350 board (520 KB internal
# SRAM, 16 MB external QSPI flash via the canonical W25Q128 SOIC-8). No PSRAM,
# no second QSPI bus.
#
#   ~5.3 M parameters
#   ~5.3 MB int8 weight blob (incl. embeddings, per-channel scales, LN gains)
#   ~384 KB KV cache + ~55 KB activations/scratch = ~440 KB SRAM at runtime
#   (~80 KB headroom under the 520 KB internal SRAM cap)
#
# Dataset: data/wedding/{train.bin, val.bin, meta.pkl} produced by
#          data/wedding/{fetch_corpus, train_tokenizer, prepare}.py
#          ~6.6 M BPE tokens at vocab_size=2048, with <eos> at every
#          paragraph boundary.

out_dir = "out-wedding"
eval_interval = 500
eval_iters = 100
log_interval = 25
always_save_checkpoint = False  # only checkpoint when val improves
wandb_log = False

dataset = "wedding"
gradient_accumulation_steps = 4
batch_size = 64
block_size = 128  # context window — comfortable for short quotes + sonnets

# Architecture — locked for RP2350 single-chip-plus-flash target
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1
bias = False

# Optim — cosine decay, AdamW betas tuned to nanoGPT defaults.
# max_iters is halved vs the 64-ctx version because each iter now processes
# 2x as many tokens — total tokens-seen stays roughly constant.
learning_rate = 3e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 3e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 500

# nanoGPT auto-detects vocab_size from data/<dataset>/meta.pkl when present
# — our prepare.py writes vocab_size=2048 there, so we don't override here.

# Tweak these on the host you're training on:
device = "cuda"   # set to 'cpu' or 'mps' for laptop dry-runs
compile = True    # set False on first run to surface errors faster
