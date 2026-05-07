"""
Train a byte-level BPE tokenizer on the wedding corpus.

Produces:
    tokenizer.json   the full HuggingFace tokenizers state (load with
                     `Tokenizer.from_file(...)`)
    meta.pkl         {vocab_size, vocab, id_to_token, special_tokens}
                     consumed by prepare.py and export_weights.py
    tokenizer_preview.txt   sample encode/decode round-trips for sanity

Usage:
    cd training
    uv run python data/wedding/train_tokenizer.py \\
        --vocab-size 2048 --min-frequency 2

Design notes:
- Byte-level BPE (the GPT-2 / RoBERTa style). Robust to any UTF-8 input,
  including the curly quotes and accented characters that show up in
  Gutenberg texts.
- Vocab size of 2048 is the firmware budget: at n_embd=256, int8, that's
  2048 * 256 = 512 KB of `wte` (counted twice via weight tying for the LM
  head, so 0.5 MB total — comfortable inside 16 MB QSPI flash).
- Special tokens are reserved at the *start* of the vocab so their IDs are
  small, fixed, and easy to test for in the firmware sampler.
- Style markers in the corpus look like `<<SONNET>>` — we add those exact
  forms as special tokens too, so a single token covers each style switch
  in the training stream.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import ByteLevel as ByteLevelProcessor

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.txt"
OUT_TOKENIZER = HERE / "tokenizer.json"
OUT_META = HERE / "meta.pkl"
OUT_PREVIEW = HERE / "tokenizer_preview.txt"

# Reserved at the start of the vocab. Order matters — these IDs are baked
# into the firmware sampler.
SPECIAL_TOKENS = [
    "<pad>",
    "<eos>",
    "<bos>",
    # Style controls. Match the `<<STYLE>>` form already written into the
    # corpus by fetch_corpus.py so that each marker becomes exactly one token.
    "<<SHAKESPEARE>>",
    "<<SONNET>>",
    "<<AUSTEN>>",
    "<<BRONTE>>",
    "<<VICTORIAN_POET>>",
    "<<ROMANTIC_POET>>",
    "<<DICKINSON>>",
    "<<WHITMAN>>",
    "<<RUMI>>",
    "<<FREEFORM>>",
    # Reserved for future use / future fine-tuning datasets.
    "<<VOW>>",
    "<<TOAST>>",
]


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)
    return tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    if not CORPUS.exists():
        print(f"corpus not found: {CORPUS}\nrun fetch_corpus.py first.",
              file=sys.stderr)
        return 1

    print(f"Training BPE on {CORPUS}")
    print(f"  vocab_size:    {args.vocab_size}")
    print(f"  min_frequency: {args.min_frequency}")
    print(f"  special tokens ({len(SPECIAL_TOKENS)}): {SPECIAL_TOKENS}")

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train([str(CORPUS)], trainer=trainer)
    tok.save(str(OUT_TOKENIZER))

    vocab = tok.get_vocab()  # token -> id
    id_to_token = {v: k for k, v in vocab.items()}
    actual_vocab_size = len(vocab)

    meta = {
        "vocab_size": actual_vocab_size,
        "vocab": vocab,
        "id_to_token": id_to_token,
        "special_tokens": {
            name: vocab[name] for name in SPECIAL_TOKENS if name in vocab
        },
    }
    with OUT_META.open("wb") as f:
        pickle.dump(meta, f)

    # Round-trip sanity check on a handful of corpus lines.
    samples = [
        "Shall I compare thee to a summer's day?\n",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune,\n",
        "How do I love thee? Let me count the ways.\n",
        "<<SONNET>>\nLove looks not with the eyes, but with the mind;\n",
    ]
    with OUT_PREVIEW.open("w", encoding="utf-8") as preview:
        preview.write(f"vocab_size = {actual_vocab_size}\n")
        preview.write(f"special_tokens = {meta['special_tokens']}\n\n")
        for s in samples:
            enc = tok.encode(s)
            dec = tok.decode(enc.ids)
            preview.write(f"INPUT:  {s!r}\n")
            preview.write(f"IDS:    {enc.ids}\n")
            preview.write(f"TOKENS: {enc.tokens}\n")
            preview.write(f"DECODE: {dec!r}\n")
            preview.write("\n")

    print()
    print(f"Wrote {OUT_TOKENIZER}")
    print(f"Wrote {OUT_META}")
    print(f"Wrote {OUT_PREVIEW}")
    print(f"actual vocab size: {actual_vocab_size}")
    print(f"special token ids: {meta['special_tokens']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
