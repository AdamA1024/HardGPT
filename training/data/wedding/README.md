# Wedding corpus + tokenizer pipeline

Three scripts, run in order, prepare everything `nanoGPT/train.py` needs.

```
fetch_corpus.py    pulls 35 public-domain works from Project Gutenberg
                   (Shakespeare, Austen, Brontes, Romantic / Victorian poets,
                   Whitman, Dickinson, Burns, FitzGerald, Wilde, Hugo, Forster).
                   Strips PG headers/footers, tags each work with a style
                   marker, writes:
                     raw/<id>.txt
                     corpus.txt              ~18 MB, ~3.3 M words
                     corpus_manifest.json

train_tokenizer.py trains a 2048-vocab byte-level BPE on corpus.txt with 15
                   reserved special tokens (<pad>, <eos>, <bos>, plus 12
                   style markers). Writes:
                     tokenizer.json
                     meta.pkl                {vocab_size, vocab, id_to_token,
                                              special_tokens}
                     tokenizer_preview.txt   round-trip samples

prepare.py         tokenizes corpus.txt with the trained BPE, writes:
                     train.bin               uint16 ids, ~6.6 M tokens
                     val.bin                 uint16 ids, ~67 k tokens (1 %)
                   and extends meta.pkl with split sizes.
```

## Reproduce

```bash
cd training
uv sync
uv run python data/wedding/fetch_corpus.py
uv run python data/wedding/train_tokenizer.py
uv run python data/wedding/prepare.py
```

A symlink `nanoGPT/data/wedding -> ../../data/wedding` makes `train.py`
pick up the dataset when invoked from inside `nanoGPT/`.

## Train

CPU smoke test (a few seconds):

```bash
cd training/nanoGPT
uv run --project .. python train.py config/train_wedding.py \
    --device=cpu --compile=False \
    --max_iters=3 --eval_iters=2 --eval_interval=2 \
    --batch_size=2 --block_size=64 \
    --n_layer=2 --n_head=2 --n_embd=64
```

Real training on a single A100 / 4090:

```bash
cd training/nanoGPT
uv run --project .. python train.py config/train_wedding.py
```

Default config: 6 layers, 8 heads, 256 embd, 256 ctx, vocab 2048, 60 K iters,
cosine 3e-4 -> 3e-5, batch 64 x grad-accum 4. Expected runtime ~30-60 min on
A100, ~1-2 h on 4090. Initial loss should be `ln(2048) = 7.62`; healthy
end-of-training val loss is roughly 3.5-4.0.

## Style imbalance — read this before launching the long run

`fetch_corpus.py` reports words per style. The current corpus is heavily
imbalanced (e.g. `freeform` ~730 k, `sonnet` ~18 k). Two reasonable
mitigations:

1. **Inflate underrepresented styles** in the corpus before tokenizing — e.g.
   duplicate the sonnets section 8x. Cheapest fix.
2. **Sampled training** — modify `train.py`'s `get_batch` to draw segments
   weighted by their style tag rather than uniformly. Cleaner but needs a
   small patch.

Defer this until after a baseline model has converged so we know which styles
look bad.
