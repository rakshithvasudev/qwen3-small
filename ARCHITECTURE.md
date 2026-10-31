# How the Qwen3_small setup came about

This document traces how we got from the reference Qwen3-0.6B to the model we trained, and why
each step went the way it did. `DECISIONS.md` is the per-choice reference: each choice with its
citation, venue and peer-review status. `report.ipynb` has every experiment's tables and curves.
`deliverables.ipynb` shows the final model against the task specification.

---

## 1. The trained model, in one paragraph

**Qwen3_small** is the Qwen3 architecture, scaled down:
- 24 layers, width 640;
- 10 query heads sharing 5 key/value heads, head dim 64;
- SwiGLU feed-forward with hidden size 1,728;
- RMSNorm pre-norm plus QK-Norm, RoPE with base 10,000;
- input embedding and output head tied, an 8,192-token byte-level BPE vocabulary, 512-token context;
- **114.4M parameters**.

It was trained with Muon at lr 4e-3, weight decay 1.0 and a warmup-stable-decay schedule, for
10 epochs over 50,000 TinyStories (97M tokens seen) on an Apple M4 Max. Result on the
5,000-story validation split: **1.574 nats/token, 0.550 bits per byte**.

```
token ids ─► Embedding 8192×640 (tied) ─► 24 × [ x += Drop(GQA(RMSNorm(x)))      ] ─► RMSNorm ─► Head 640×8192 (= Embedding)
                                                [ x += Drop(SwiGLU(RMSNorm(x)))   ]
GQA: Q 640→10×64, K,V 640→5×64, RMSNorm on q and k, RoPE(base 1e4), causal SDPA, out 640→640
SwiGLU: fc3( silu(fc1 x) * fc2 x ), 640→1728→640, no biases
```

---

## 2. The starting point and the fixed constraints

The task fixes four things:

1. **Architecture:** start from the reference Qwen3 implementation (rasbt, LLMs-from-scratch,
   `ch05/11_qwen3/standalone-qwen3.ipynb`) and modify it into a "Qwen3_small" of about
   100–200M parameters. A different model of that size is not acceptable.
2. **Data:** 10,000–50,000 TinyStories, split 90/10 or 80/20.
3. **Outputs:**
   - the loss falling step by step, with at least two validation evaluations;
   - generations from prompts such as "Once upon a time" at temperatures 0.5 / 0.8 / 1.0 / 1.2, up to 30 tokens;
   - a sentiment classifier built on this model, graded against `emotions_classified.csv`.
4. **Hardware:** one laptop. Apple M4 Max with 48 GB unified memory and a 40-core GPU, running
   PyTorch on the MPS backend, with a cooling break between runs.

The reference Qwen3-0.6B, measured with our own code:

| | Qwen3-0.6B |
|---|---|
| Parameters | 596.0M (155.6M of it the embedding, 26%) |
| Width / layers | 1024 / 28 |
| Heads (Q:KV) / head dim | 16:8 / 128 |
| SwiGLU hidden | 3,072 |
| Vocabulary | 151,936 |
| Context / RoPE base | 40,960 / 1,000,000 |

It is three to six times too large. Everything below is about how to shrink it, and about what
"best" means once it is shrunk.

---

## 3. The fact that shaped everything: we have very little data

50,000 stories, after tokenizing the 45,000 training stories, come to **9.7M tokens**.

- **The ratio:** a 100–200M model on that is roughly **0.05–0.1 tokens per parameter**.
  Compute-optimal training (Hoffmann et al., NeurIPS 2022) uses about 20.
- **What that implies:** every source we checked agrees the model is oversized for the data.
  - The BabyLM 10M-word track was won by models of 24–30M parameters (CoNLL 2023 / 2024).
  - Repeating data stays almost as useful as fresh data for about 4 epochs, and stops helping
    by about 16 (Muennighoff et al., NeurIPS 2023).

That gives four rules that the rest of the design follows:

1. **Stay near the bottom of the allowed range.** Extra parameters will mostly memorise.
2. **Use the maximum data the task allows** (50,000 stories), and train for many epochs.
3. **Regularise hard**: weight decay and dropout, with early stopping on validation loss.
4. **Treat published results on shape and optimizer as hypotheses to test, not facts.** None of
   them were measured this far from compute-optimal.

---

## 4. Shrinking the vocabulary first

The first thing to cut is not the transformer. It is the vocabulary.

- **The embedding cost:** with Qwen's 151,936-token vocabulary at width 640, the embedding table
  alone is **97.2M parameters**. Our final 109M transformer body plus that table would be
  **206M, over the 200M limit**.
- **What the extra vocabulary would buy:** almost nothing. TinyStories uses roughly 1,500 distinct
  words, so most of those 151k rows would never be trained.
- **Speed and memory:** in an early benchmark, the 151k-way output layer took about 63% of all
  training FLOPs. At batch 64 the logits tensor exceeded an MPS kernel limit (more than 2³¹
  elements) and the run failed.

So we trained our own byte-level BPE on the training split only.
- **Byte-level** means any text, including the classifier CSV, encodes with no unknown tokens (tested).
- **Normalisation:** text is lowercased and apostrophes are removed, the same way for pretraining
  and for the classifier.

The vocabulary scaling law of Tao et al. (NeurIPS 2024) predicts a small optimum here, roughly
8–10k. We tested 8k against 16k on equal data, compared on **bits per byte**, because loss per
token isn't comparable across tokenizers:

| Vocab | Params | Val bits/byte | Speed |
|---|---|---|---|
| 8,192 | 110.9M | 0.67158 | 10.3k tok/s |
| 16,384 | 115.6M | 0.67158 | 9.9k tok/s |

The two tied to five decimals. Doubling the vocabulary only shortens TinyStories by 0.6%, because
8k merges already cover nearly every word. **8k won** on parameters and speed.

The classifier CSV tokenizes slightly better with 16k: 1.12 vs 1.21 tokens per word. That only
matters if texts are truncated, so the classifier's maximum length was set to 96 tokens. The
longest CSV text is 88 tokens, so nothing is cut.

---

## 5. Context length and positions

- **Context: 40,960 → 512.**
  - 97.2% of stories fit in 512 tokens; at 256, 18% would be split.
  - Stories are packed back to back with `<eos>` separators, so a long story continues into the
    next window rather than being cut off.
  - Attention cost grows with the square of context, so there is no reason to go longer.
- **RoPE base: 1,000,000 → 10,000.** Qwen3's large base exists to spread rotation frequencies
  over 40k-token contexts. At 512 tokens the standard base of 10,000 is the usual choice.

---

## 6. Choosing the shape: from one guess to six testable hypotheses

### The first candidate (147M), and why it was replaced

The first pass of research proposed a single model:
- width 768, 18 layers, SwiGLU 2,560, 16k vocabulary, about **147M parameters**;
- it was benchmarked at about 10.1k tokens/s, which made it feasible on this laptop.

It was never trained. The proportions were sensible, but no particular result supported that
exact shape, so it couldn't be defended any better than many alternatives. Two things also cut
against it:
- §3 argues for the bottom of the size range;
- §4 showed that a large vocabulary wastes parameters.

### Six presets, each taken from a published result

We replaced the single guess with six shapes. Each one is the configuration a specific paper says
should win at this size. The first four are matched at 110–115M, so they test depth against width
at nearly equal size. The last two test whether spending extra budget helps.

| Preset | Width / layers / Q:KV / SwiGLU | Params | The claim it tests | Source |
|---|---|---|---|---|
| `mobilellm` | 576 / 30 / 9:3 / 1,536 | 110.9M | Deep and thin wins at ~125M (the MobileLLM-125M / SmolLM2-135M shape) | ICML 2024; COLM 2025 |
| `levine` | 640 / 24 / 10:5 / 1,728 | 114.4M | Depth helps up to an optimum near 24 layers at this width | NeurIPS 2020 |
| `deepnarrow` | 512 / 36 / 8:4 / 1,408 | 110.4M | Gains from depth saturate at 32–36 layers | ICLR 2022 |
| `classic12` | 768 / 12 / 12:4 / 3,072 | 110.1M | 12 layers is optimal around 134M (the GPT-2 / Pythia depth) | NAACL 2024 |
| `wide8` | 1024 / 8 / 16:4 / 4,096 | 130.0M | Shape barely matters at fixed size; width stores facts | arXiv 2020; arXiv 2023 (not peer-reviewed) |
| `mlpheavy` | 640 / 18 / 10:5 / 3,456 | 146.8M | Feed-forward width should grow fastest (the old 147M candidate's neighbour) | NeurIPS 2023 (vision) |

Shared choices:
- **Head dim 64,** so heads × head dim equals the width.
- **About 2 query heads per KV head,** as in Qwen3-0.6B.
- **SwiGLU hidden size near 2.7× width** for the matched presets, following MobileLLM / SmolLM2.

**Prediction, written down before running:** the deep shapes do at least as well as `classic12`,
which beats `wide8`, and every gap is small, because data rather than shape is the bottleneck.

### Throughput first: depth isn't too slow

Before the quality comparison, we timed all six (bf16 autocast, `torch.compile`, 512-token sequences):

| Preset | tok/s |
|---|---|
| `classic12` | 15.9k |
| `wide8` | 14.4k |
| `deepnarrow` | 12.4k |
| `levine` | 12.2k |
| `mlpheavy` | 11.9k |
| `mobilellm` | 11.3k |

Thirty layers cost at most ~30% in speed against twelve. That is acceptable, so every shape stayed in.

### The result

All six were trained on the same 9.7M tokens (50k stories, 1 epoch, Muon lr 4e-3):

| Preset | Params | Val bits/byte |
|---|---|---|
| **`levine`** | 114.4M | **0.6669** |
| `deepnarrow` | 110.4M | 0.6714 |
| `mobilellm` | 110.9M | 0.6720 |
| `mlpheavy` | 146.8M | 0.6743 |
| `classic12` | 110.1M | 0.6775 |
| `wide8` | 130.0M | 0.6883 |

- **The prediction held:** the deep shapes beat 12 layers, which beat 8, with a 3% spread from best to worst.
- **Extra parameters didn't help:** the two larger models, 130M and 147M, lost to the 110M deep ones.
- **Seed noise:** rerunning `levine` with another seed gave 0.6706, so seed alone shifts the
  result by about ±0.004. The top three are tied within that noise.
- **Why `levine` anyway:** it has the best measured result in that tied group, its shape comes
  from a peer-reviewed analysis of depth against width, and it runs as fast as `mobilellm`.
- **What we claim from this:** deep beats shallow-and-wide at ~110M. We do not claim that
  24 layers beats 30.

---

## 7. What we kept from Qwen3, and what we changed in the code

**Kept unchanged:**
- SwiGLU feed-forward without biases;
- RMSNorm pre-norm (eps 1e-6) and a final norm;
- grouped-query attention;
- QK-Norm;
- RoPE;
- tied input and output embeddings;
- the reference class names and structure: `Qwen3Model`, `TransformerBlock`,
  `GroupedQueryAttention`, `FeedForward`, `RMSNorm`, `compute_rope_params`, `apply_rope`.

QK-Norm in particular stays on purpose. It prevents attention logits from growing at high
learning rates (Wortsman et al., ICLR 2024), and no run diverged, up to lr 8e-3.

**Implementation changes, none of them a new architectural mechanism:**

| Change | Why |
|---|---|
| Attention through PyTorch's fused `scaled_dot_product_attention` (`is_causal`, `enable_gqa`) | Same maths as the reference's explicit softmax (unit-tested), much faster on MPS |
| Residual dropout 0.1 | Ten passes over a small dataset; dropout helps when repeating data (Xue et al., NeurIPS 2023). Off at inference |
| GPT-2-style initialisation: N(0, 0.02), residual output projections scaled by 1/√(2·layers) | The reference only loads pretrained weights. Training from scratch needs a small init, or the tied embedding gives huge initial logits |
| fp32 weights with bf16 autocast | Stable optimizer state; bf16 speed in the matrix multiplies |

---

## 8. Training recipe: what we learned, in order

The sweep used short proxy runs to set the recipe before spending hours on the final runs. Each
stage fixed one thing and handed it to the next. Every run was followed by a 10-minute cooling
break; long runs also paused for 10 minutes every 45 minutes.

### Stage 1: learning rate, and a warmup surprise

Setup: `mobilellm`, 20k stories, 1 epoch (236 steps).
- **First pass:** Muon was far ahead of AdamW, 2.34 vs 3.01 validation loss.
- **But AdamW's curve was wrong:** at lr ≥ 1e-3 it sat on the "unigram plateau" (loss ~5.6) from
  step 20 to 60.
- **The cause:** 5% warmup of a 236-step run is only 11 steps.
- **The fix:** with 20% warmup (47 steps), AdamW reached 2.81 at both 5e-4 and 1e-3.
- **Muon's optimum, 4e-3,** was confirmed by testing either side: 2e-3 and 8e-3 were both worse.
- **What we took forward:**
  - at least 10% warmup in every later proxy;
  - AdamW at 5e-4 (tied with 1e-3; take the lower one, because optimal LRs fall as runs get longer);
  - Muon at 4e-3.

### Stage 2: vocabulary (§4) and stage 3: shape (§6)

These selected **8k** and **`levine`**. Both used Muon, the optimizer the final model was most
likely to use. Muon's update size is matched to AdamW's, which is what lets one learning rate
transfer across the different widths.

### Stage 4: weight decay, the single biggest effect

Setup: `levine`, 20k stories, 6 epochs.

| wd | Best val loss | What happened |
|---|---|---|
| 0.1 | 2.048 | Memorised. Validation rose after ~3.8 epochs while train fell to 0.87 |
| 0.5 | 1.864 | Still improving at the end |
| **1.0** | **1.822** | Still improving at the end |
| 2.0 | 1.870 | Too much |

- The usual value of 0.1 is far too low here. The best value, 1.0, sits inside the range
  tested, since 2.0 was worse.
- This matches Kim et al. (ICLR 2026): data-constrained, multi-epoch training wants much higher
  weight decay, about 0.8 at 150M.

### Stage 5: final runs, AdamW against Muon at full length

Setup: `levine`, 8k vocabulary, wd 1.0, 50k stories, up to 10 epochs. Each optimizer used its usual
schedule:

| | AdamW + cosine | Muon + WSD |
|---|---|---|
| Learning rate | 5e-4 | 4e-3 |
| Best checkpoint | epoch 5.9 (early-stopped at 6.7) | epoch 10 (last step) |
| Full validation loss | 1.731 | **1.574** |
| Bits per byte | 0.605 | **0.550** |
| Train − val gap at best | 0.33 | 0.10 |

**An early-stopping bug, found and fixed during this stage:**
- The first Muon run stopped at step 3,500. Under WSD the learning rate stays constant for 80% of
  the run, and the loss sits on a flat plateau. Five evaluations of small wobbles there triggered
  the early stop, before the ramp-down.
- The fix: early stopping now only counts evaluations during the ramp-down for WSD (and
  throughout for cosine). The stopped run is kept in `runs/final_attempt1/` as a reference.

**How to read the result:**
- **Muon's gain came almost entirely at the end.** It sat at about 2.07 for 80% of the run, with
  train equal to validation (heavily regularised, not overfitting). The ramp-down then took it
  from 2.07 to 1.59.
- **At equal steps before that, AdamW was ahead** (1.78 vs 2.08 at step 3,000). The short stage-1
  runs had overstated Muon's per-token advantage, as Wen et al. (ICLR 2026) warn about short
  benchmarks.
- **AdamW began overfitting at epoch ~6.** Muon got more out of ten passes over the same data.
- **What this compares:** each optimizer paired with its usual schedule. It does not isolate the
  optimizer, and the cross-pairings (AdamW + WSD, Muon + cosine) weren't run.

**The Muon model was selected as the final model.**

---

## 9. From language model to sentiment classifier

- **Model:** the pretrained Qwen3_small plus a linear layer on the hidden state of the last
  non-padding token.
  - Attention is causal, so the last token is the only position that has seen the whole
    sentence, and padding added after it cannot affect it (unit-tested).
  - Mean pooling was run as an alternative.
- **Data:**
  - 31 duplicate texts removed;
  - a stratified 80/10/10 split;
  - model selection on validation macro-F1;
  - the test split used once.
- **Label:** graded on the CSV's `sentiment` column; the 6-class `emotion` column is a secondary result.

| Sentiment, held-out test split | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.549 | 0.354 |
| TF-IDF + logistic regression | 0.939 | 0.938 |
| Qwen3_small, frozen, only the head trained | 0.722 | 0.720 |
| Same architecture, **untrained**, fine-tuned | 0.947 | 0.947 |
| Qwen3_small, pretrained, fine-tuned, mean pooling | 0.967 | 0.967 |
| **Qwen3_small, pretrained, fine-tuned, last token** | **0.974** | **0.974** |

- **Pretraining is worth 2.7 points** over the same architecture fine-tuned from scratch, and the
  pretrained model is 3.5 points ahead of TF-IDF.
- **The untrained model's 0.947 and TF-IDF's 0.939** show that much of this task can be solved
  from keywords.
- **The frozen probe is weak** because the model learned to continue children's stories, not to
  represent this kind of text. Its features need fine-tuning.
- **On the 6-class `emotion` column:** fine-tuned Qwen3_small reaches **0.928 accuracy / 0.904
  macro-F1**, against TF-IDF's 0.866 / 0.841. The rarest classes are the hardest: surprise 0.85 F1,
  love 0.87.
- **Last-token pooling beats mean pooling** by 0.6 points (one standard error on 1,597 test texts
  is ~0.4), a weak preference that fits the causal-attention argument.

---

## 10. How it is checked

`uv run pytest` covers:
- **The model:**
  - the reference class names exist;
  - every preset falls in 100–200M;
  - input and output weights are tied;
  - causality: changing a later token does not change earlier logits;
  - fused attention equals the reference maths;
  - a tiny batch can be overfit.
- **Data and tokenizer:**
  - a 90/10 split with no story in both halves;
  - the tokenizer round-trips with no unknown tokens.
- **Generation:** follows the spec's output format.
- **Classifier:**
  - duplicates removed, a stratified split with no leakage;
  - the backbone equals the pretrained checkpoint;
  - padding has no effect.
- **The final runs, against the spec:**
  - parameter count and story count;
  - a split ratio of 0.9;
  - a training loss at every step;
  - at least two validation evaluations, with validation loss falling;
  - a saved checkpoint, and generations at all four temperatures.

---

## 11. Known limitations

- **Encoding errors in the data.** 5.6% of training stories contain mis-decoded curly quotes from
  the source data (e.g. `â€™`). The model occasionally reproduces them at high temperature. It was
  found mid-sweep and left alone, so the stages stay comparable. Fix:
  `s.encode("cp1252").decode("utf-8")` in `normalize()`.
- **Single seed** for every stage except one shape check. Differences under ~0.004 bits/byte are noise.
- **Proxy runs** (1 epoch, or 20k stories) selected the vocabulary, shape and weight decay. The
  final runs were 10 epochs on 50k stories, and the optimizer comparison shows that short-run
  rankings can change at full length.
- **Optimizer and schedule are confounded** in the final comparison (§8).
- **Muon's best checkpoint was its last step,** so a longer run might improve it further. Not tested.
