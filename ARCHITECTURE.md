# Architecture & Wedding Adaptation Plan

This document captures the existing HardGPT architecture (so we can fork it
intentionally) and lays out the implementation plan for repurposing the project
into a wedding / romance / literary quote generator.

The original artifact: a 3-layer character-level transformer running entirely
on a TI MSPM0G3507 Cortex-M0+ MCU, generating fake-Shakespeare on a 16x2 LCD.
Recipients of the wedding-version artifact: an uncle (senior ASIC designer at
Broadcom) and an aunt (graduate admissions officer at Stanford). The "AI
running on a PCB" novelty must be preserved; capability should expand because
hardware constraints are now relaxed.

---

## 1. Existing System Architecture

### 1.1 Hardware

| Block | Part | Role |
|---|---|---|
| MCU | TI **MSPM0G3507** (ARM Cortex-M0+, 80 MHz, 128 KB flash, 32 KB SRAM, **no FPU, no DSP/SIMD**) | Runs the entire transformer in software |
| Display | 16x2 HD44780 LCD on a PCF8574 I2C backpack (`0x27`), 4-bit interface | Char-by-char "typewriter" output |
| I/O | 3 push buttons (Generate/Temp/Seed, active-low, internal pull-ups), 3 LEDs (running/tick/confidence), one active piezo through a 2N3904 NPN switch | UI and "alive" indicators |
| Power | AMS1117 LDO from 5 V to 3.3 V, 1N4148 reverse-pol diode, 47 uF bulk, slide switch | Single-rail; STANDBY0 clock policy |
| Programming | SWD via PA19 (SWDIO) / PA20 (SWCLK) | Flashed from CCS over an XDS-class probe |
| Build | Hand-etched single-sided PCB, all THT (resistors, LEDs, push-buttons, sockets, pin headers) | Aesthetic artifact, not just functional |

Pin map lives in `hardgpt.syscfg` (TI SysConfig); the SDK regenerates
`ti_msp_dl_config.{c,h}` from it during a CCS build.

### 1.2 Model architecture

Tiny GPT-2 / nanoGPT topology, no biases, weight-tied LM head:

| Param | Value |
|---|---|
| `N_LAYER` | 3 |
| `N_HEAD` | 4 |
| `N_EMBD` | 48 |
| `HEAD_DIM` | 12 |
| `BLOCK_SIZE` | 48 tokens |
| `VOCAB_SIZE` | 65 (chars from tiny-shakespeare) |
| Activation | approximate GELU |
| Norm | LayerNorm (no bias) |
| Output head | tied to `wte` |
| Trained params | ~95 K (decode-time int8 ~88 KB) |

### 1.3 Tokenizer / vocabulary

Pure character-level. `prepare.py`
(`training/nanoGPT/data/shakespeare_char/`) downloads tiny-shakespeare from the
char-rnn repo, sorts unique characters, and builds `stoi/itos` dictionaries.
The 65-char alphabet is `\n !$&',-.3:;?A-Za-z`. The vocab table is dumped
verbatim into `weights.h` as `const char vocab[65]`. There is no detokenizer
beyond `vocab[token_id]`.

### 1.4 Inference pipeline (on-chip)

`main.c` is a single translation unit, top to bottom:

1. **Embedding lookup** — `forward()` computes
   `x[i] = wte_q*scale + wpe_q*scale` for the current token / position
   (no matmul, just a dequant accumulator).
2. **3x `block_forward`**, each doing:
   - `LayerNorm` -> `quantize_vec` -> **int8 x int8 -> int32 -> float matvec**
     (`matvec_i8i8_pc`) for QKV.
   - Quantize new K and V into `kv_cache[layer][pos]` with their own
     per-vector scales.
   - Per-head attention against `pos+1` cached K,V vectors (dequant on the
     fly), `1/sqrt(head_dim)` scaling, softmax, weighted-V accumulation.
   - Output projection -> residual.
   - LayerNorm -> 4x MLP (matmul, GELU, matmul) -> residual.
3. **Final LayerNorm**, then **logits = wte . x_norm** reusing the embedding
   table (weight tying).
4. **Sampling**: temperature divide -> softmax -> categorical via 32-bit LCG
   (`1664525, 1013904223`). Greedy fallback when `temp <= 0`.

`slide_kv_cache()` shifts everything left by one slot once
`pos == BLOCK_SIZE`, so the model perpetually sees the most recent 47 tokens.

Math primitives are deliberately ugly because there is no FPU: Quake
`0x5f3759df` invsqrt + one Newton step, Schraudolph
`12102203 * x + 1064866805` exp bit-hack, Pade `x(27+x^2)/(27+9x^2)` tanh.

### 1.5 Memory layout

Static globals in SRAM:

- `x`, `x_residual`, `x_norm` (3 * 48 * 4 B = 576 B)
- `qkv` (576 B), `ffn_hidden` (768 B), `logits` (260 B), `x_q` (192 B)
- **`kv_cache[3][48][96]` = 13,824 B**, plus `kv_scales[3][48][2]` = 1,152 B
- `attn_scores[48]` = 192 B
- Stack + SDK globals + LCD buffers

Total active RAM ~17-18 KB out of 32 KB. Flash is dominated by the int8
weight blob in `weights.h` (~88 KB), plus float LayerNorm gains (~600 B), plus
the .text/.rodata of CCS + driverlib.

### 1.6 Firmware stack

- TI **MSPM0 SDK 2.10.00.04** (`driverlib`)
- TI **SysConfig** generates `ti_msp_dl_config.{c,h}` into `Debug/`
- TI **Code Composer Studio (CCS)** as the IDE/build/flash environment
- Single `main.c` + header-only `lcd.h` + generated `weights.h`
- No RTOS, single super-loop with busy-wait button polling

### 1.7 Training / export pipeline

```
input.txt -- prepare.py --> train.bin / val.bin + meta.pkl
                                   |
nanoGPT/train.py + config/train_micro.py
                                   v
                      out-micro/ckpt.pt  (PyTorch)
                                   |
                      export/export_weights.py
                                   v
                          export/weights.h
                                   | (manual cp)
                                   v
                    ../weights.h    (firmware build input)
```

Training is `uv`-managed (`pyproject.toml`: torch >= 2.11, numpy, tqdm,
requests, tiktoken — though tiktoken is currently unused). `train_micro.py`
runs 30 K iters of AdamW at lr 1e-3 -> 1e-4 with warmup, batch 64, dropout
0.2, on `device='mps'`.

`export_weights.py` does **per-output-channel symmetric int8 quantization**
(one scale per row of every 2-D weight, one per row of `wte`/`wpe`); LayerNorm
gains stay float. It writes a single C header with the architecture
`#define`s, the vocab table, and every weight matrix as `const int8_t name[N]`
plus `const float name_scales[M]`.

### 1.8 PCB / power & dataflow

Single-sided through-hole layout (etched). 5 V in -> AMS1117 -> 3.3 V rail.
PA0/PA1 carry I2C up to the LCD (4-line interface multiplexed through the
PCF8574 plus a backlight bit). Buttons short PA5/PA7/PA8 to GND with internal
pull-ups doing the rest. The buzzer is driven through Q (2N3904) so the MCU
pin only sources base current. The DIP-32 socket is the LP-MSPM0G3507
LaunchPad-style breakout — the dev kit sits in the socket so the artisanal
etched board hosts but doesn't replace it.

### 1.9 Bottlenecks and constraints in the original

| Bottleneck | Why it bites |
|---|---|
| **128 KB flash** | Caps total int8 weights at ~95 KB, which is what forces the 3-layer / 48-dim / 65-vocab box. |
| **32 KB SRAM** | KV cache scales linearly with `N_LAYER * BLOCK_SIZE * N_EMBD`; doubling any of those instantly busts SRAM. |
| **No FPU / no SIMD on M0+** | Every multiply is software-emulated; matmul is the bottleneck. The Schraudolph exp and Quake invsqrt are mandatory, not stylistic. |
| **Char-level vocab** | Each token carries < 2 bits of useful information; 48-token context = ~50 chars ~= 8 words. Coherent multi-sentence output is structurally impossible. |
| **`block_size = 48`** | Even with longer training, the model can't form sentences longer than its context window without forgetting how it started. |
| **Greedy / pure-temperature sampling** | No top-k, top-p, or repetition penalty; long runs degrade into repeats. |
| **16x2 LCD** | 32 visible chars; readers can't see a full sentence at once. |
| **Single `weights.h` triplicated** | The same 96 KB blob lives in three places (`weights.h`, `training/inference_c/weights.h`, `training/export/weights.h`). Easy to drift. |
| **CCS / SysConfig coupling** | Build is not reproducible from CLI without TI proprietary tooling. |

---

## 2. External Dependencies

| Category | Dependency | Notes |
|---|---|---|
| **Training repo** | [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT) (vendored under `training/nanoGPT/`, MIT) | `model.py`, `train.py`, `sample.py`, `configurator.py` are upstream; `config/train_micro.py` is project-specific. |
| **Dataset** | tiny-shakespeare from `karpathy/char-rnn` (`raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`) | Downloaded on demand by `prepare.py`; not committed. |
| **Pretrained checkpoint** | `training/nanoGPT/out-micro/ckpt.pt` | **Not in repo** — must be re-trained from scratch (`uv run python nanoGPT/train.py config/train_micro.py`). |
| **Generated meta** | `training/nanoGPT/data/shakespeare_char/{train.bin, val.bin, meta.pkl}` | Produced by `prepare.py`; required by `export_weights.py`. |
| **Generated weights** | `training/export/weights.h` and the firmware-side `weights.h` | Both regenerated; the firmware copy is a manual `cp`. |
| **Python deps** (`pyproject.toml`) | `torch>=2.11`, `numpy>=2.4`, `tqdm`, `requests`, `tiktoken` (declared but unused for char-level) | Installed via `uv sync`. Trains on MPS by default. |
| **Embedded SDK** | TI **MSPM0 SDK** 2.10.00.04 (`@ti/driverlib`, SysConfig) | Headers + generated `ti_msp_dl_config.{c,h}` are not in this repo; pulled in by CCS at build time. |
| **IDE / build / flash** | TI **Code Composer Studio**, TI **SysConfig**, XDS-class debug probe (LP-MSPM0G3507 onboard XDS110 works) | The `targetConfigs/MSPM0G3507.ccxml` is auto-generated by CCS. |
| **PCB toolchain** | **KiCad >= 9.0** (project format `version 20250114`) | Footprints from KiCad standard libs only (no custom symbols). |
| **Manufacturing** | Hand-etched (toner transfer / photoresist) — not a fab order; THT components only, hand-soldered | The etched-PCB photo in `img/etched-pcb.png` is the artifact. |
| **Implicit / not in repo** | `input.txt` (downloaded), `ckpt.pt` (trained), `Debug/` build outputs, the LP-MSPM0G3507 LaunchPad (sits in the DIP-32 socket) | Reproduce or supply locally. |

---

## 3. Wedding Quote Project — Adaptation Plan

The brief: keep the soul of the artifact (a transformer materially running on
an etched PCB, generating text in real time, no internet) while loosening
hardware constraints enough that the output actually feels like a wedding gift
instead of a text-generation toy.

### 3.0 Headline architectural choices

| Decision | Choice | Why |
|---|---|---|
| MCU | **RP2350B** (Cortex-M33 dual-core w/ FPU + DSP, 520 KB SRAM, 16 MB QSPI flash, optional PSRAM) | Keeps the bare-metal, "AI on a PCB" novelty; the FPU+DSP gets us roughly 10x the existing inference rate without rewriting math primitives. Hand-solderable QFN-80, $1 chip — a Broadcom ASIC engineer will appreciate that it's an open silicon datasheet. **Alternative: ESP32-S3-WROOM-1U-N16R8** if PSRAM-backed larger model matters more than FPU. |
| Tokenizer | **2,048-entry custom BPE** trained on the romance corpus | Char-level can't produce coherent vows in 48 tokens of context. BPE at 2K keeps `wte` ~256 KB at `n_embd=128, int8`, fits comfortably and dramatically lifts output quality per inference step. |
| Model | **6 layers, 8 heads, 256 embd, 256 ctx** (~5 M int8 params, ~5 MB) | Sized to fit inside 16 MB QSPI flash with room for assets, and small enough that single-token latency stays human-readable on a 150 MHz M33 with FPU. |
| Display | **2.42" 128x64 SSD1309 OLED** (or 2.13" e-paper alternative) | Lets a whole sonnet line fit on screen; white-on-black looks like engraved type. Etched-PCB + OLED + brass standoffs reads as a wedding object, not a hobbyist demo. |
| Persistence of novelty | The PCB still does all inference itself; **no Wi-Fi, no host, no cloud** | The artifact's value depends on it being genuinely self-contained — never compromise on this. |

### 3.1 Dataset strategy

Build a single corpus of out-of-copyright English love/wedding text (Project
Gutenberg sources):

- **Shakespeare**: complete sonnets + Romeo & Juliet, A Midsummer Night's
  Dream, Much Ado, Twelfth Night.
- **Austen**: Pride and Prejudice, Sense and Sensibility, Emma, Persuasion,
  Mansfield Park, Northanger Abbey.
- **Bronte**: Jane Eyre, Wuthering Heights.
- **Romantic / Victorian poets**: Elizabeth Barrett Browning (Sonnets from the
  Portuguese), Keats, Shelley, Burns, Christina Rossetti, Wordsworth,
  Tennyson.
- **American**: Whitman (Leaves of Grass — selected), Dickinson (love poems).
- **Translations safely PD**: FitzGerald's Rubaiyat.

Curate to ~5-15 MB of UTF-8 text. Tag each segment with a one-token style
prefix (`<sonnet>`, `<vow>`, `<austen>`, `<whitman>`, `<rumi>`) so the model
can be steered at inference time.

**Stage two (optional, post-MVP)**: build a small *quotation* corpus by
extracting 1-4-sentence excerpts that look like deliverable quotes, and
**fine-tune** on that subset so generations end at quote-shaped boundaries.

### 3.2 Tokenizer updates

Train a **byte-level BPE** with `tokenizers` or `sentencepiece` to a
**2,048-entry** vocabulary on the assembled corpus. Reserve indices 0-7 for
special tokens: `<pad>`, `<eos>`, `<sonnet>`, `<vow>`, `<austen>`, `<whitman>`,
`<rumi>`, `<freeform>`.

Port the tokenizer encode/decode to C as a flash-resident structure:

- `vocab_strings[]` — packed UTF-8 bytes
- `vocab_offsets[2048]` — into the byte pool
- A **trie** or sorted array for greedy longest-match decode of byte streams.
  Encode side is only needed if we ever take user input — keep tiny.

The on-device path only needs **decode** (id -> bytes), so the firmware
tokenizer table is < 16 KB.

### 3.3 Fine-tuning / retraining approach

Train **from scratch on the pooled corpus** rather than fine-tuning anything
large — distillation from a frontier model is feasible but unnecessary at this
scale and would muddy the "every weight on this board was trained for this
gift" story.

1. `prepare_wedding.py` (replaces `data/shakespeare_char/prepare.py`):
   downloads Gutenberg texts, normalizes whitespace, applies the BPE, splits
   95/5 train/val, writes `train.bin / val.bin / meta.pkl`.
2. `config/train_wedding.py` (replaces `train_micro.py`):
   - `n_layer=6, n_head=8, n_embd=256, block_size=256, dropout=0.1, bias=False`
   - `batch_size=64, grad_accum=4, max_iters=200_000, lr=3e-4 -> 3e-5 cosine,
     warmup=2_000`
   - Optionally enable **rotary position embeddings** in `model.py` instead of
     learned `wpe` — eliminates the `wpe` table and lets the model extrapolate
     past `block_size` cleanly.
3. After convergence, **fine-tune for 5-10 K steps on the curated short-quote
   subset** to bias the model toward quote-shaped outputs.
4. Save best-by-val checkpoint to `out-wedding/ckpt.pt`.

Train on a single rented A10/A100 hour or local 4090 — converges in
single-digit hours.

### 3.4 Inference / runtime changes

Replace the C-header weight blob with a **flat binary** (`weights.bin`) loaded
from a fixed flash address; the firmware mmaps it. Reasons: 5 MB of
`const int8_t` literals stalls the compiler and bloats build artifacts.

| Change | Reason |
|---|---|
| Add `arm_math.h` (CMSIS-DSP) and replace `matvec_i8i8_pc` with `arm_dot_prod_q15`-style int8/int16 inner loops | M33 SIMD-MAC instructions ~= 4x speedup |
| Replace `fast_invsqrt` / `fast_expf` with `vsqrtf` and a small range-reduction `expf` (FPU is now native) | Cleaner code; numerical accuracy improves and the bit-hacks no longer help |
| Add **top-k + top-p (nucleus)** sampling and a **repetition penalty** (subtract alpha from logits of recently sampled tokens) | Char-level temperature sampling masks how thin the distribution is; with 2K-vocab this matters a lot |
| Add **prompt prefix** support: when style button picks "Sonnet," seed with `<sonnet>` token before generation | Style steering without retraining |
| Make `BLOCK_SIZE = 256` and keep KV cache as int8 | Real sentence-length context |
| Hoist the inference loop onto **core 1** of RP2350; UI / display / button polling stays on core 0 | Display refreshes don't block generation |
| Stream output token-by-token through a small detokenizer FIFO so that BPE's multi-byte tokens render to the OLED smoothly | Otherwise you'd see partial glyphs |

### 3.5 Firmware stack changes

- Drop CCS + SysConfig + MSPM0 SDK; switch to **pico-sdk** (`cmake` +
  `arm-none-eabi-gcc`), reproducible from any shell and matches the
  open-silicon vibe.
- Re-organize `main.c` into: `main.c` (boot/UI), `model.c/.h` (forward pass),
  `tokenizer.c/.h`, `display.c/.h`, `sampler.c/.h`.
- Move weight loading to a `weights_loader.c` that maps the flash region
  containing `weights.bin` and exposes the same `Layer` struct table.
- Replace busy-wait button polling with **GPIO IRQ + 20 ms debounce timer** on
  core 0. (Buttons currently lose presses during a `forward` call.)

### 3.6 Hardware modifications

| Subsystem | Change | Note |
|---|---|---|
| MCU | RP2350B QFN-80 footprint replaces the LP-MSPM0G3507 socket | Bring out USB D+/D-, SWD, 12 MHz crystal pads, QSPI flash, optional QSPI PSRAM |
| Flash | 16 MB W25Q128 | 5 MB weights + assets + firmware |
| PSRAM | 8 MB IPS6404 (optional) on the second QSPI CS | Only needed if KV cache / activations grow past 520 KB SRAM |
| Display | 2.42" SSD1309 OLED breakout via SPI (4-wire SPI: SCK, MOSI, CS, DC, RES) | Or 2.13" GxEPD2-class e-paper for the wedding-stationery aesthetic |
| Buttons | 3 -> 4 push buttons: **Generate / Style / Save / Heart** | "Save" persists the last quote to flash; "Heart" lights an extra LED and bumps a counter |
| Indicators | 3 LEDs preserved; one upgraded to a **WS2812** driven from a single GPIO so it can pulse pink during generation | Preserves the "alive" feel without adding board area |
| Audio | Replace active piezo with a small **passive piezo + PWM** | Lets you play a 4-note motif at start/end instead of a flat beep |
| Power | USB-C input (CC1/CC2 5.1 kOhm pulldowns), TPS62203 buck -> 3.3 V, optional CR2032 + diode-OR | Consumer-friendly; AMS1117 dropout was fine but a buck is cooler under load |
| Programming | USB-C BOOTSEL button (RP2350 native) replaces SWD-only flashing | Recipients can update firmware without a probe |
| Real-time clock | Tiny 32 kHz crystal on RP2350's RTC pins | Lets the device greet "Happy 1st anniversary" on the wedding date |

### 3.7 PCB modifications

- Re-target to **KiCad 9** project, larger board outline (~100 x 80 mm) —
  gives room for engraved silkscreen art (names, date, heart motifs) without
  crowding parts.
- Stay single-sided **etchable** — that's the whole aesthetic. Verify
  routability with the new BOM by relaxing trace widths to 0.4 mm and using a
  couple of hand-soldered jumper wires for crossings (the RP2350's QFN may
  need a small breakout / interposer; ESP32-S3-WROOM-1U is module-based and
  avoids fine-pitch QFN entirely — that's the best argument for the ESP32
  path if hand-etching the QFN-80 footprint is too tight).
- Add a back-of-board **dedication zone** (silkscreen): names, wedding date,
  "fork of HardGPT," credit line for the corpus authors and Karpathy.
- Add 4x M3 mounting holes plus brass standoff footprints so the board can be
  displayed standing up in a frame.

**Tradeoff to flag**: if hand-etching a QFN-80 RP2350 is uncomfortable, drop
in an **RP2350-Stamp** or **Pimoroni Pico Plus 2** as a castellated module
sitting in a header — same MCU, no fine-pitch soldering, slight loss of the
"every part on the board" purity.

### 3.8 UX / display / input changes

- **Boot**: 4-note motif, OLED shows a heart and "<NameA> ♥ <NameB> — press
  Generate."
- **Idle**: cycles through { name, wedding date, "press Generate" } with a
  slow fade.
- **Generate**: streams the BPE-decoded token bytes to the OLED with a 4-row
  scrolling buffer and a soft cursor. Word wrap on space.
- **Style** button: cycles `<sonnet> / <vow> / <austen> / <whitman> / <rumi> /
  <freeform>`, label shown in 8-pt font in the top-left.
- **Save**: writes the last completed quote (max 1 KB) to a 64 KB ring buffer
  in flash; long-press shows saved quotes.
- **Heart** button: toggles a slow pink pulse and increments an EEPROM-backed
  counter.
- **End-of-quote handling**: stop generation when the model emits `<eos>` or
  after a max of e.g. 220 BPE tokens (~150 words), play a soft 2-note motif.
- **No internet, no app, no companion screen** — the artifact is the entire
  experience. This is the gift.

### 3.9 Memory / storage changes

| Region | Use | Approx size |
|---|---|---|
| QSPI flash 0x10000000+ | RP2350 firmware (.text/.rodata) | ~300 KB |
| QSPI flash | `weights.bin` (model + tokenizer table + style embeddings) | ~5-6 MB |
| QSPI flash | Saved-quote ring buffer | 64 KB |
| QSPI flash | Asset blob (boot art, icon font) | ~64 KB |
| Internal SRAM | Activations + small KV cache | 128-256 KB |
| Optional PSRAM | Full KV cache @ ctx=256, l=6 | 768 KB |

`weights.bin` layout: 16-byte header (magic, version, n_layer, n_head,
n_embd, block_size, vocab_size), then per-tensor records `{tag, dtype, shape,
scales_offset, data_offset}`, then a packed data section. This makes the
firmware tolerant to architecture tweaks without recompiling.

### 3.10 Deployment / testing steps

1. **Tokenizer + corpus**: `fetch_corpus.py` runs end-to-end, produces
   `corpus.txt`. `train_tokenizer.py` produces `tokenizer.json` + `meta.pkl`.
   `prepare.py` produces `train.bin/val.bin`. Sanity check vocab coverage on
   hand-picked passages.
2. **Train on a single GPU** (~few hours), monitor val loss and sample
   qualitatively at 25 %, 50 %, 100 % of training.
3. **Export**: `export_wedding.py` (extends the existing `export_weights.py`
   to emit a flat `weights.bin` instead of a `weights.h`, plus to embed the
   tokenizer table).
4. **Desktop reference (`training/inference_c/`)**: extend the existing
   reference impl with the new model dims, BPE decode, top-k/top-p sampling.
   Run side-by-side with the PyTorch `sample.py` and confirm bit-for-bit
   logits agreement (modulo quantization noise) on a fixed seed.
5. **Bring-up board**: spin a small dev board with just the RP2350 + flash +
   USB-C + OLED + one button. Validate inference latency and OLED rendering
   before committing to the artistic full board.
6. **Full board**: hand-etch, populate, smoke-test 3.3 V rail, flash a blink,
   then progressively bring up OLED, buttons, audio, and finally inference.
7. **Quote-quality QA**: run 1,000 generations per style, score for:
   grammaticality, length, no profanity (filter list as last-line defense), no
   obvious repetition. Iterate fine-tuning until pass rate is acceptable.
8. **Field test**: leave the device generating for 24 h on battery; confirm no
   memory leaks (long-run-stable on bare metal usually means: confirm
   KV-cache slide is correct and the flash-ring-buffer wrap works).
9. **Final pass**: dedication silkscreen, brass standoffs, a small wooden
   frame, a printed card explaining how to use it.

---

## 4. Risks to flag now

1. **Hand-etched QFN-80 is tight.** Mitigation: drop in an RP2350-Stamp or
   Pimoroni Pico Plus 2 module — zero fine-pitch soldering, all the
   capability. If you want every component to be discrete, plan for a small
   fab order (~$5/board JLCPCB) and brand it as "etched dedication plate +
   production logic board."
2. **Tokenizer in C is the most error-prone subsystem** — get the desktop
   reference impl producing identical output to the Python tokenizer on a
   10K-sample harness *before* moving on.
3. **Style-token steering can collapse** if the corpus is unbalanced — keep
   per-style token counts within ~2x of each other or upweight underrepresented
   styles during training.
4. **Don't let the project drift into "make it a chat box"** — the
   AI-on-a-PCB novelty depends on it being a single-purpose gift, not a
   Wi-Fi-enabled smart speaker.
