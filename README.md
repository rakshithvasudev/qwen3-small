# Qwen3_small

A 114M-parameter Qwen3, pretrained from scratch on 50k TinyStories on an Apple M4 Max (MPS),
then fine-tuned as a sentiment classifier graded on `emotions_classified.csv`.

| | Result |
|---|---|
| Model | Qwen3 architecture (GQA, QK-Norm, SwiGLU, RMSNorm, RoPE, tied embeddings), 24 layers × 640, 8k BPE vocab, 114.4M params |
| Training | Muon lr 4e-3, warmup-stable-decay schedule, weight decay 1.0, 10 epochs over 45k train stories (97M tokens) |
| Validation (5k held-out stories) | **1.574 nats/token, 0.550 bits/byte** (AdamW + cosine baseline: 1.731 / 0.605) |
| Sentiment classifier (held-out test, n = 1,597) | **97.4% accuracy** (same model from random init: 94.7%; TF-IDF + LR: 93.9%) |

## Where to look

| File | What it is |
|---|---|
| `deliverables.ipynb` | The task specification, item by item, run on the final model |
| `report.ipynb` | Every experiment stage: tables and loss curves |
| `ARCHITECTURE.md` | How the setup came about, as a narrative |
| `DECISIONS.md` | Every choice with the evidence for it (paper, venue, peer-review status) and what our runs showed |

## Setup

```bash
uv sync
uv run --with datasets python fetch_data.py   # first 105k TinyStories -> data/ts105k.jsonl
# put emotions_classified.csv in artifacts/spec/ (needed by classify.py, baselines.py and the tests)
uv run pytest                                  # 39 tests; the final-run checks skip until runs/final/ exists
```

The tokenizer and the tokenized datasets are built and cached in `data/` on first use.

## Quick demos

**Loss falling, from scratch** (~3 minutes; prints train loss every step, val loss every 20 steps,
then a sample):

```bash
uv run python train.py --name demo --config levine --stories 10000 --epochs 1 \
  --optim muon --lr 4e-3 --sched wsd --warmup-frac 0.1 --wd 1.0 --eval-every 20 --max-steps 60
```

Val loss starts at ~9.13 (≈ ln 8192, a uniform guess) and reaches ~3.8 by step 40. `--compile`
makes steps faster but adds about a minute of compilation before the first one.

**Generation from the trained model** (temperatures 0.5 / 0.8 / 1.0 / 1.2, max 30 new tokens):

```bash
uv run python generate.py --ckpt runs/final/muon/best.pt
```

**Classifier:**

```bash
uv run python baselines.py                                   # majority class and TF-IDF + LR
uv run python classify.py --ckpt runs/final/muon/best.pt     # full fine-tune, sentiment
uv run python classify.py --ckpt runs/final/muon/best.pt --label emotion
```

## Reproducing the experiments

Each stage runs a grid of `train.py` runs one after another, with a 10-minute cooling break between
runs. Results go to `runs/<stage>/<run>/` and `runs/<stage>/summary.csv`. Stages, in the order run:

| # | Stage | Command | Picks |
|---|---|---|---|
| 0 | Throughput of six shapes | `uv run python sweep.py throughput` | — |
| 1 | Learning rate | `uv run python sweep.py lr --config mobilellm --lrs 2.5e-4 5e-4 1e-3 2e-3 4e-3 8e-3`, then `--warmup-frac 0.2 --lrs 5e-4 1e-3 --only adamw_lr0.0005_wu0.2 adamw_lr0.001_wu0.2` | Muon 4e-3, AdamW 5e-4 |
| 2 | Vocab 8k vs 16k | `uv run python sweep.py vocab --optim muon --lr 4e-3 --config mobilellm` | 8k |
| 3 | Shape | `uv run python sweep.py shapes --optim muon --lr 4e-3 --vocab 8192` | `levine` |
| 4 | Weight decay | `uv run python sweep.py wd --optim muon --lr 4e-3 --config levine --wds 0.1 0.5 1.0 2.0` | 1.0 |
| 5 | Final AdamW vs Muon | `uv run python sweep.py final --config levine --lr 5e-4 --muon-lr 4e-3 --wd 1.0 --epochs 10` | Muon |
| 6 | Classifier ablations | `classify.py` with `--mode full/probe/random`, `--pool last/mean`, `--label sentiment/emotion` | full, last token |
| 7 | Qwen3-MoE variants | `uv run python sweep.py moe --optim muon --lr 4e-3` | rejected (ties dense with 70% more params) |
| 8 | Final recipe on 100k stories | `uv run python sweep.py final100k --config levine --lr 4e-3 --wd 1.0 --epochs 10` | better LM (1.490), same classifier |

The shape seed check is `train.py --name shapes_seed/levine_seed1 --config levine --seed 1` with
the stage-3 settings. Stage 8 is above the 10k–50k story range of the task, so the 50k model
remains the deliverable.

## Layout

```
qwen3_small/   model.py (Qwen3Model, dense and MoE feed-forward), config.py (presets),
               data.py (normalisation, 90/10 split, packing), tokenizer.py (byte-level BPE)
train.py       pretraining loop (AdamW or Muon, cosine or WSD, bf16 autocast, early stopping)
sweep.py       experiment stages
generate.py    sampling in the specification's output format
classify.py    classifier fine-tuning; baselines.py: majority and TF-IDF references
fetch_data.py  TinyStories download
tests/         model, data, MoE, generation, classifier and final-run checks
```
