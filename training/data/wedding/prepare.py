"""
Tokenize the wedding corpus with the trained BPE and write `train.bin`,
`val.bin`, and `meta.pkl` in the format `nanoGPT/train.py` expects.

Usage:
    cd training
    uv run python data/wedding/prepare.py

Inputs:
    data/wedding/corpus.txt          (from fetch_corpus.py)
    data/wedding/tokenizer.json      (from train_tokenizer.py)
    data/wedding/meta.pkl            (from train_tokenizer.py — extended here)

Outputs:
    data/wedding/train.bin           uint16 token ids
    data/wedding/val.bin             uint16 token ids
    data/wedding/meta.pkl            extended with split sizes

Format choice — uint16 means our tokenizer must have <= 65536 tokens. Our
target is 2048, so this is comfortable and matches the existing
`data/shakespeare_char` convention.

Split: 99/1 train/val. The corpus is ~3.3 M words / ~18 MB; even 1% gives a
val set of ~30 k tokens, plenty for a stable val-loss signal.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.txt"
TOKENIZER = HERE / "tokenizer.json"
META_PATH = HERE / "meta.pkl"
TRAIN_BIN = HERE / "train.bin"
VAL_BIN = HERE / "val.bin"

VAL_FRACTION = 0.01


def main() -> int:
    if not CORPUS.exists():
        print(f"corpus not found: {CORPUS}", file=sys.stderr)
        return 1
    if not TOKENIZER.exists():
        print(f"tokenizer not found: {TOKENIZER}\n"
              f"run train_tokenizer.py first.", file=sys.stderr)
        return 1

    print(f"Loading tokenizer from {TOKENIZER}")
    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab_size = tok.get_vocab_size()
    print(f"  vocab_size: {vocab_size}")
    if vocab_size > 65535:
        print("  vocab_size > 65535 — uint16 storage will overflow.",
              file=sys.stderr)
        return 2

    print(f"Loading corpus from {CORPUS} "
          f"({CORPUS.stat().st_size/1024/1024:.2f} MB)")
    text = CORPUS.read_text(encoding="utf-8")

    print("Encoding...")
    encoded = tok.encode(text)
    ids = encoded.ids
    print(f"  total tokens: {len(ids):,}")

    arr = np.asarray(ids, dtype=np.uint16)
    n = len(arr)
    n_val = max(1024, int(n * VAL_FRACTION))
    n_train = n - n_val
    train_ids = arr[:n_train]
    val_ids = arr[n_train:]

    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    print(f"  train: {len(train_ids):,} tokens -> {TRAIN_BIN.name}")
    print(f"  val:   {len(val_ids):,} tokens -> {VAL_BIN.name}")

    meta = {}
    if META_PATH.exists():
        with META_PATH.open("rb") as f:
            meta = pickle.load(f)
    meta.update({
        "vocab_size": vocab_size,
        "tokenizer_path": str(TOKENIZER.relative_to(HERE.parent.parent)),
        "n_train_tokens": int(len(train_ids)),
        "n_val_tokens": int(len(val_ids)),
    })
    with META_PATH.open("wb") as f:
        pickle.dump(meta, f)

    print(f"Updated {META_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
