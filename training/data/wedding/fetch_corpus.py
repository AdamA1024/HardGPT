"""
Download a curated corpus of public-domain romantic / wedding / literary text
from Project Gutenberg, strip the PG headers and footers, tag each work with a
style label, and write:

    raw/<id>.txt           one file per source (post-strip, UTF-8)
    corpus.txt             concatenated, style-tagged corpus for tokenizer + LM
    corpus_manifest.json   {gutenberg_id, title, author, style, chars, words}

Usage:
    cd training
    uv run python data/wedding/fetch_corpus.py

Notes:
- Polite to PG: we send a real user-agent, sleep between requests, and prefer
  the stable cache/epub mirror. If a download fails we skip and keep going.
- Style tags are inserted as `\n<<STYLE>>\n` once at the start of each work.
  Training code can either keep these as plain text or remap them to special
  token IDs after BPE training.
- Files are cached under raw/ so reruns don't re-download.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_CORPUS = HERE / "corpus.txt"
OUT_MANIFEST = HERE / "corpus_manifest.json"

UA = (
    "GPTweddingCorpusFetcher/0.1 "
    "(+https://github.com/canukguy03/GPTwedding; private gift project)"
)
SLEEP_SEC = 1.5  # be polite

# Project Gutenberg "cache/epub" URLs are the most stable. Fallback to "files".
URL_CACHE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
URL_FILES = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
URL_FILES2 = "https://www.gutenberg.org/files/{id}/{id}.txt"


@dataclass
class Source:
    id: int
    title: str
    author: str
    style: str  # one of: sonnet, vow, austen, bronte, whitman, dickinson,
                # romantic_poet, victorian_poet, rumi, shakespeare, freeform


# Curated list. Style tags drive prompt-conditioning during training and
# generation. IDs are Project Gutenberg eBook numbers; comment shows the
# expected title for sanity-checking.
SOURCES: list[Source] = [
    # --- Shakespeare ---
    Source(1041, "Shakespeare's Sonnets",          "William Shakespeare", "sonnet"),
    Source(1513, "Romeo and Juliet",               "William Shakespeare", "shakespeare"),
    Source(1514, "A Midsummer Night's Dream",      "William Shakespeare", "shakespeare"),
    Source(1519, "Much Ado About Nothing",         "William Shakespeare", "shakespeare"),
    Source(1526, "Twelfth Night",                  "William Shakespeare", "shakespeare"),
    Source(1524, "Hamlet",                         "William Shakespeare", "shakespeare"),
    Source(1112, "The Tempest",                    "William Shakespeare", "shakespeare"),

    # --- Austen ---
    Source(1342, "Pride and Prejudice",            "Jane Austen", "austen"),
    Source(161,  "Sense and Sensibility",          "Jane Austen", "austen"),
    Source(158,  "Emma",                           "Jane Austen", "austen"),
    Source(105,  "Persuasion",                     "Jane Austen", "austen"),
    Source(141,  "Mansfield Park",                 "Jane Austen", "austen"),
    Source(121,  "Northanger Abbey",               "Jane Austen", "austen"),

    # --- Bronte ---
    Source(1260, "Jane Eyre",                      "Charlotte Bronte", "bronte"),
    Source(768,  "Wuthering Heights",              "Emily Bronte", "bronte"),
    Source(767,  "Agnes Grey",                     "Anne Bronte", "bronte"),

    # --- Romantic / Victorian poets ---
    Source(2002, "Sonnets from the Portuguese",    "Elizabeth Barrett Browning", "victorian_poet"),
    Source(23684,"Endymion",                       "John Keats", "romantic_poet"),
    Source(2490, "Lamia, Isabella, ...",           "John Keats", "romantic_poet"),
    Source(4800, "The Complete Poetical Works",    "Percy Bysshe Shelley", "romantic_poet"),
    Source(8861, "Poems",                          "William Wordsworth", "romantic_poet"),
    Source(19221,"Goblin Market, The Prince's Progress, and Other Poems",
                                                   "Christina Rossetti", "victorian_poet"),
    Source(8601, "Idylls of the King",             "Alfred, Lord Tennyson", "victorian_poet"),
    Source(13619,"The Princess",                   "Alfred, Lord Tennyson", "victorian_poet"),

    # --- American ---
    Source(1322, "Leaves of Grass",                "Walt Whitman", "whitman"),
    Source(12242,"Poems by Emily Dickinson, Series One",
                                                   "Emily Dickinson", "dickinson"),
    Source(12243,"Poems by Emily Dickinson, Series Two",
                                                   "Emily Dickinson", "dickinson"),
    Source(12241,"Poems by Emily Dickinson, Series Three",
                                                   "Emily Dickinson", "dickinson"),

    # --- Burns / Scottish ---
    Source(9863, "Poems and Songs",                "Robert Burns", "romantic_poet"),

    # --- Translated, public domain ---
    Source(246,  "Rubaiyat of Omar Khayyam",       "Omar Khayyam (tr. FitzGerald)", "rumi"),
    Source(2400, "The Mathnawi (Book I)",          "Jalalu'd-Din Rumi (tr. Whinfield)", "rumi"),

    # --- Romance / wedding-flavoured prose, public domain ---
    Source(135,  "Les Miserables",                 "Victor Hugo (tr. Hapgood)", "freeform"),
    Source(2641, "A Room with a View",             "E. M. Forster", "freeform"),
    Source(174,  "The Picture of Dorian Gray",     "Oscar Wilde", "freeform"),  # for romantic-aesthetic prose
    Source(844,  "The Importance of Being Earnest","Oscar Wilde", "freeform"),
]


PG_START_PATTERNS = [
    re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I),
    re.compile(r"\*END\*THE SMALL PRINT.*?\*END\*", re.I | re.S),
]
PG_END_PATTERNS = [
    re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I),
    re.compile(r"End of (?:the )?Project Gutenberg(?:'s)?[^\n]*\n", re.I),
]


def strip_pg_boilerplate(text: str) -> str:
    """Remove the PG header/footer surrounding the actual book text."""
    # Find a start marker and trim everything before it (inclusive).
    start_idx = 0
    for pat in PG_START_PATTERNS:
        m = pat.search(text)
        if m:
            start_idx = max(start_idx, m.end())
    text = text[start_idx:]

    # Find an end marker and trim from there.
    end_idx = len(text)
    for pat in PG_END_PATTERNS:
        m = pat.search(text)
        if m:
            end_idx = min(end_idx, m.start())
    text = text[:end_idx]

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of more than 2 blank lines to exactly 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace.
    return text.strip() + "\n"


def fetch(source: Source, session: requests.Session) -> str | None:
    """Download a PG book; cache locally; return stripped text or None."""
    cache_path = RAW_DIR / f"{source.id}.txt"
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        return cache_path.read_text(encoding="utf-8")

    for url_template in (URL_CACHE, URL_FILES, URL_FILES2):
        url = url_template.format(id=source.id)
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"  request error for {url}: {e}", file=sys.stderr)
            continue
        if r.status_code != 200:
            print(f"  {r.status_code} from {url}", file=sys.stderr)
            continue
        # PG sometimes serves Latin-1; try utf-8 first, then latin-1.
        raw = r.content
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        stripped = strip_pg_boilerplate(text)
        if len(stripped) < 1024:
            print(f"  suspiciously small after strip ({len(stripped)} B), skipping",
                  file=sys.stderr)
            continue
        cache_path.write_text(stripped, encoding="utf-8")
        time.sleep(SLEEP_SEC)
        return stripped

    return None


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    manifest = []
    total_chars = 0
    total_words = 0

    with OUT_CORPUS.open("w", encoding="utf-8") as out:
        for src in SOURCES:
            print(f"[{src.id:>5}] {src.author:.30s} | {src.title:.50s} ({src.style})")
            text = fetch(src, session)
            if text is None:
                print("  FAILED — skipping")
                continue

            chars = len(text)
            words = len(text.split())
            total_chars += chars
            total_words += words

            # Tag the work with its style. Plain-text marker for now;
            # the tokenizer step can remap to a single special token.
            out.write(f"\n<<{src.style.upper()}>>\n")
            out.write(f"# {src.title} — {src.author}\n\n")
            out.write(text)
            out.write("\n")

            manifest.append({
                **asdict(src),
                "chars": chars,
                "words": words,
            })

    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"Wrote {OUT_CORPUS}")
    print(f"  total works:  {len(manifest)}")
    print(f"  total chars:  {total_chars:,}")
    print(f"  total words:  {total_words:,}")
    print(f"  on-disk size: {OUT_CORPUS.stat().st_size/1024/1024:.2f} MB")

    # Per-style breakdown
    by_style: dict[str, int] = {}
    for m in manifest:
        by_style[m["style"]] = by_style.get(m["style"], 0) + m["words"]
    print()
    print("Words per style:")
    for style, n in sorted(by_style.items(), key=lambda x: -x[1]):
        print(f"  {style:18s} {n:>10,}")

    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
