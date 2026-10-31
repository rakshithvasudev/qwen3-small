# Design decisions

Each entry records the choice, what was rejected, the evidence it rests on (with venue and
peer-review status), and what our own runs showed. Each stage's raw table is in
`runs/<stage>/summary.csv`; `ARCHITECTURE.md` tells the same story as a narrative.

Hardware: Apple M4 Max, 48 GB unified memory, 40-core GPU, PyTorch MPS backend.

## 0. The regime we are in

The spec fixes the model at 100–200M parameters and the data at 10k–50k TinyStories. 50k stories
are ~9.7M training tokens (8k BPE), i.e. **~0.09 tokens per parameter**, against ~20 for
compute-optimal training (Hoffmann et al., NeurIPS 2022). Every source agrees this model is
oversized for the data. For example, the BabyLM 10M-word track was won by models of 24–30M
parameters (BabyLM findings, CoNLL 2023 / 2024, peer-reviewed).

So the general strategy is:
- stay near the bottom of the allowed size range;
- use all 50k stories and train for many epochs;
- regularize (weight decay, dropout) and use early stopping on validation loss.

Muennighoff et al. (NeurIPS 2023, peer-reviewed) found that up to ~4 epochs of repeated data
is nearly as good as fresh data, with value decaying to near zero by ~16 epochs (R*≈16).

None of the papers below tested exactly this regime, so their claims are treated as hypotheses
and checked against our runs.

## 1. Architecture: modified Qwen3, not another model

The reference is rasbt's `standalone-qwen3.ipynb` (LLMs-from-scratch, ch05/11_qwen3). The class
structure is kept in `qwen3_small/model.py`, and so are the Qwen3 parts:
- grouped-query attention;
- QK-RMSNorm;
- SwiGLU feed-forward;
- pre-norm RMSNorm;
- RoPE;
- tied embeddings.

What changed:

| Change | Reason |
|---|---|
| Width/depth/heads scaled to ~111M (see §3) | Spec range 100–200M |
| Vocab 151,936 → 8,192 (own BPE, see §2) | With 151k vocab the embedding table alone is ~87M params at d=576, and the LM head is ~63% of FLOPs at this size; it also overflows an MPS kernel limit (tensor > INT_MAX) at batch 64 |
| Context 40,960 → 512 | 97% of stories fit (8k tokenizer); 256 would split ~18% of them. Stories are packed with `<eos>`, so a longer story continues into the next window rather than being cut off |
| head_dim 128 → 64; heads 16:8 → 10:5 | Keeps n_heads × head_dim = emb_dim (640) at the smaller width; 2 query heads per KV head, the same GQA ratio as Qwen3-0.6B |
| RoPE base 1e6 → 1e4 | 1e6 is for 40k-token contexts; 1e4 is the standard choice at 512 |
| Attention via `F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)` | Same math as the reference (unit-tested), fused kernel |
| Residual dropout 0.1 | Multi-epoch small-data training; dropout helps when repeating data (Xue et al., NeurIPS 2023, peer-reviewed) |
| Init N(0, 0.02), residual projections scaled 1/√(2L) | GPT-2 style; needed for stable 30-layer training |

QK-norm is kept on purpose: it prevents attention-logit growth at high LR (Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities", ICLR 2024 oral). No run has diverged so far (up to lr 4e-3 for both optimizers).

## 2. Tokenizer: own byte-level BPE, 8k vocab (16k tested)

- **Rejected: Qwen's 151k tokenizer.** It wastes most of the parameter budget on embeddings that
  TinyStories' ~1.5k-word vocabulary never uses, and it fails on MPS at batch 64.
- **Evidence:** Tao et al., "Scaling Laws with Vocabulary" (NeurIPS 2024, peer-reviewed). Their
  compute-optimal vocabulary for ~100M non-embedding parameters is far below 151k. Extrapolating
  their formula to our data size favours roughly 8–10k.
- **Details:**
  - trained on the training split only (no validation leakage);
  - byte-level, so no `<unk>` is possible, which matters for the classifier CSV (tested);
  - text lowercased and apostrophes stripped, for both pretraining and the classifier.
- **Comparison across vocab sizes:** bits-per-byte, because nats/token are not comparable when
  tokens differ in length.
- **Our result** (stage 2: `mobilellm`, Muon lr 4e-3, 50k stories × 1 epoch):

  | Vocab | Params | Val loss (nats/token) | Val bpb | Val tokens | CSV tokens/word | tok/s |
  |---|---|---|---|---|---|---|
  | 8,192 | 110.9M | 1.9233 | 0.67158 | 1,085,183 | 1.205 | 10.3k |
  | 16,384 | 115.6M | 1.9347 | 0.67158 | 1,078,787 | 1.120 | 9.9k |

  - Bits-per-byte is tied to five decimals. The 16k vocabulary shortens TinyStories by only 0.6%,
    because a ~1.5k-word vocabulary is already almost fully covered by 8k merges.
  - The CSV tiebreak favours 16k slightly (7% fewer tokens per word).
  - That only matters if texts get truncated, so the classifier's max length was raised to 96
    tokens (the longest CSV text is 88 tokens at 8k) and nothing is cut.
  - **Chosen: 8k.** Equal quality, 4.7M fewer parameters (all of them in the embedding), 4% faster.
    This matches Tao et al.'s prediction that the optimal vocabulary here is small.

## 3. Model shape: six paper-derived presets, ~110–150M

| Preset | d / L / heads (Q:KV) / FFN | Params (8k) | Claim tested | Venue |
|---|---|---|---|---|
| `mobilellm` | 576 / 30 / 9:3 / 1536 | 110.9M | Deep-and-thin wins at ~125M; this is the MobileLLM-125M / SmolLM2-135M shape | MobileLLM: ICML 2024; SmolLM2: COLM 2025 |
| `levine` | 640 / 24 / 10:5 / 1728 | 114.4M | Depth-efficiency optimum near L≈24 at this width | Levine et al., NeurIPS 2020 |
| `deepnarrow` | 512 / 36 / 8:4 / 1408 | 110.4M | Gains from depth saturate around 32–36 layers | Tay et al. (DeepNarrow), ICLR 2022 |
| `classic12` | 768 / 12 / 12:4 / 3072 | 110.1M | 12 layers optimal around 134M | Petty et al., NAACL 2024 |
| `wide8` | 1024 / 8 / 16:4 / 4096 | 130.0M | Shape matters little at fixed params / width stores facts | Kaplan et al., arXiv 2020 (not peer-reviewed); TinyStories, arXiv 2023 (ICLR submission withdrawn) |
| `mlpheavy` | 640 / 18 / 10:5 / 3456 | 146.8M | MLP width should grow fastest | Alabdulmohsin et al. (ViT shape), NeurIPS 2023 (vision) |

The first four are matched at 110–115M, which makes a clean depth-vs-width test; the last two
test spending extra budget. Pre-registered expectation: the deep shapes do at least as well as
`classic12`, which beats `wide8`, with small gaps, since data rather than shape is the bottleneck.

Throughput (tokens/s, bf16 autocast + `torch.compile`, seq 512, 8k vocab):

| Preset | tok/s |
|---|---|
| `classic12` | 15.9k |
| `wide8` | 14.4k |
| `deepnarrow` | 12.4k |
| `levine` | 12.2k |
| `mlpheavy` | 11.9k |
| `mobilellm` | 11.3k |

Depth costs at most ~30% in wall-clock.

**Our result** (stage 3: Muon lr 4e-3, WSD, 8k vocab, 50k stories × 1 epoch = 9.7M tokens, equal for all shapes):

| Preset | Params | Val loss | Val bpb | Wall-clock (min) |
|---|---|---|---|---|
| `levine` | 114.4M | **1.9100** | **0.6669** | 15.9 |
| `deepnarrow` | 110.4M | 1.9227 | 0.6714 | 16.5 |
| `mobilellm` | 110.9M | 1.9245 | 0.6720 | 16.0 |
| `mlpheavy` | 146.8M | 1.9311 | 0.6743 | 17.8 |
| `classic12` | 110.1M | 1.9404 | 0.6775 | 13.6 |
| `wide8` | 130.0M | 1.9712 | 0.6883 | 15.2 |

- **The pre-registered ordering holds:** the deep shapes (24–36 layers) beat 12 layers, which
  beats 8 layers.
- **The gaps are small,** 3% from best to worst, as Kaplan et al. predict.
- **Extra parameters bought nothing.** `wide8` (+20M) and `mlpheavy` (+37M) lose to the 110M deep
  shapes. This is consistent with data, not capacity, being the limit (§0).
- **`levine` (24 layers × 640) wins** despite having fewer layers than `mobilellm` (30) or
  `deepnarrow` (36). That supports Levine et al.'s claim that the depth benefit is not
  monotonic and that the optimum near this width is around 24 layers.
- **Seed noise:**
  - Rerunning `levine` with seed 1 (different init and data order) gives 0.6706 bpb, against
    0.6669 with seed 0, so seed noise is about ±0.004 bpb.
  - The top three (`levine`, `deepnarrow`, `mobilellm`) are therefore **tied within noise**.
  - `classic12` (+0.011) and `wide8` (+0.021) are worse by more than the noise.
- **Chosen: `levine`.** It has the best measured result in a tied group, the shape is grounded in
  a peer-reviewed depth/width analysis, and it runs at the same speed as `mobilellm`.
- **What the shape result supports:** "deep beats shallow-and-wide at ~110M". It does not support
  a claim that 24 layers beats 30 or 36.

## 4. Optimizer: AdamW vs Muon

Both optimizers use the same settings:
- 16,384 tokens per step;
- gradient clipping at 1.0;
- bf16 autocast with fp32 master weights.

**AdamW:**
- β = (0.9, 0.95);
- weight decay on matrices only.

**Muon (`torch.optim.Muon`):**
- runs on all 2-D weight matrices inside the transformer blocks;
- AdamW handles the tied embedding/head and the norm gains (Jordan et al. 2024, blog; Liu et al., "Muon is scalable" / Moonlight, arXiv 2025, not peer-reviewed);
- `adjust_lr_fn="match_rms_adamw"` sets the update RMS to match AdamW's, so a single LR can be shared, following Moonlight;
- momentum 0.95, Nesterov.

**Evidence:** Wen et al., "Fantastic pretraining optimizers and where to find them" (ICLR 2026,
peer-reviewed). With every optimizer carefully tuned, Muon gives a ~1.3–1.4× speedup at 130M,
shrinking with scale. Semenov et al. (NeurIPS 2026 poster) report similar results.

**Stage 1 (LR sweep).** Setup:
- `mobilellm`;
- 20k stories, 1 epoch, i.e. 236 steps / 3.9M tokens;
- WSD schedule, wd 0.1, 5% warmup.

Best validation loss for each LR:

| lr | AdamW, 5% warmup | AdamW, 20% warmup | Muon, 5% warmup |
|---|---|---|---|
| 2.5e-4 | 2.988 | | |
| 5e-4 | 3.013 | **2.813** | 2.652 |
| 1e-3 | 3.453 | 2.823 | 2.467 |
| 2e-3 | 3.345 | | 2.367 |
| 4e-3 | 3.333 | | **2.340** |
| 8e-3 | | | 2.355 |

- **AdamW's first deficit was a warmup artifact.**
  - With 5% warmup (11 steps), AdamW at lr ≥ 1e-3 sat on the unigram plateau (train loss ~5.6) from step ~20 to ~60.
  - With 20% warmup (47 steps) the plateau disappears.
  - AdamW is then flat across 5e-4 and 1e-3 (2.81 / 2.82).
- **Muon needs no extra warmup.** Its optimum, 4e-3, is bracketed: 8e-3 is slightly worse, and nothing diverged.
- **Even with tuned warmup, Muon is clearly ahead at equal tokens:** 2.34 vs 2.81 nats (0.82 vs
  0.98 bpb). This short-horizon gap is larger than the ~1.4× speedup Wen et al. report. One
  reason: Wen et al. warn that short runs exaggerate early-training differences, so the final
  multi-epoch runs are the real test.
- **Consequences:**
  - Every later stage uses at least 10% warmup.
  - Stages 2–4 (vocab, shape, weight decay) use Muon lr 4e-3 with WSD, the optimizer the final
    model is most likely to use. Its `match_rms_adamw` scaling is designed so that one LR
    transfers across widths, which is what the shape sweep needs.
  - The final stage trains both optimizers at full length.
- **Final-run learning rates:**
  - AdamW 5e-4 (tied with 1e-3 once warmup is fixed). Muon 4e-3 (bracketed by 2e-3 and 8e-3).
  - The optimal LR tends to fall as the training horizon grows, so a short-proxy optimum is an
    upper bound. Where two LRs tie, the lower one is taken.
  - AdamW gets 5% warmup of ~5.9k steps (~300 steps), far more than the 47 steps that removed the
    plateau.

## 5. Schedule

- **Cosine** decays to 1% of peak; it is the AdamW default.
- **WSD** holds a constant LR, then runs a 1−√ cooldown over the last 20% of steps (Hägele et al.,
  "Scaling laws and compute-optimal training beyond fixed training durations", NeurIPS 2024
  spotlight; MiniCPM, COLM 2024). Its advantage is that the run length need not be fixed in
  advance; with early stopping, cosine and WSD reached similar loss in Hägele et al.
- **Final runs** pair each optimizer with the schedule it is usually run with (AdamW + cosine,
  Muon + WSD). The LR sweep used WSD for both, so the LR comparison itself is like-for-like.

## 6. Weight decay and epochs

- **Evidence:** Kim et al. (ICLR 2026 oral) find that in data-constrained, multi-epoch training
  the optimal weight decay is much higher than the usual 0.1 (~0.8 at 150M), and that it pairs
  with larger models.
- **Our test:** `levine`, Muon lr 4e-3, 20k stories × 6 epochs (1,416 steps).

  | wd | Best val loss | Best at step | Final train loss (20-step mean) |
  |---|---|---|---|
  | 0.1 | 2.048 | 900 | 0.87 |
  | 0.5 | 1.864 | 1416 (end) | 1.36 |
  | 1.0 | **1.822** | 1416 (end) | 1.61 |
  | 2.0 | 1.870 | 1416 (end) | |

  - With wd 0.1 the model memorises: validation bottoms out after ~3.8 epochs and then rises,
    while train loss falls to 0.87.
  - wd 1.0 lowers the best validation loss by 0.23 nats and is still improving at the end. This is
    the largest effect in any stage, and it confirms Kim et al. in our regime.
  - 1.0 is the edge of the grid, so wd 2.0 is tested before the final runs.
- **wd 2.0** reaches 1.870, worse than 1.0 (1.822), so **wd 1.0 is bracketed** and is used for both final runs.
- **Early stopping and WSD:**
  - The first final Muon run stopped at step 3,500 (epoch 5.9) because validation loss wobbled
    on the constant-LR plateau for 5 evals (best 2.0756 at step 3,000). It therefore never
    reached the cooldown, which is where WSD gets its drop.
  - Patience now only counts during the cooldown (last 20% of steps) for WSD, and throughout
    for cosine.
  - The stopped run is kept in `runs/final_attempt1/muon` as a no-cooldown reference.
  - Train and validation loss were level (~2.07 vs ~2.08) at that point, so the plateau was the
    LR noise floor, not overfitting.
- **Note on the coupled decay:** both optimizers apply decay as `lr × wd` per step, so wd is used
  at the same nominal value for both. For AdamW this is the setting Kim et al. tuned.
- **Final runs:**
  - up to 10 epochs;
  - early stopping with patience 5 evals;
  - best-validation checkpoint kept;
  - a 10-minute cooling pause every 45 minutes of training (excluded from the throughput numbers).

## 6b. Final runs

| Run | Schedule | Best val loss (eval subset) | Full val loss / bpb | Best at | Stopped |
|---|---|---|---|---|---|
| AdamW lr 5e-4, wd 1.0 | cosine, 10-epoch horizon | 1.7466 | **1.7314 / 0.6046** | step 3,500 (epoch 5.9) | early, step 4,000 |
| Muon lr 4e-3, wd 1.0, attempt 1 (no cooldown) | WSD, stopped in the constant phase | 2.0756 | | step 3,000 | early, step 3,500 (bug, §6) |
| Muon lr 4e-3, wd 1.0 | WSD with cooldown | 1.5887 | **1.5738 / 0.5495** | step 5,920 (epoch 10, last step) | no, ran the full 10 epochs |

**AdamW** (train loss is a 50-step mean):

| Step | Epoch | Train | Val | LR |
|---|---|---|---|---|
| 1,000 | 1.7 | 2.060 | 2.061 | 4.8e-4 |
| 2,000 | 3.4 | 1.691 | 1.850 | 4.0e-4 |
| 3,000 | 5.1 | 1.409 | 1.782 | 2.7e-4 |
| 3,500 | 5.9 | 1.419 | **1.747** | 2.0e-4 |
| 4,000 | 6.7 | 1.226 | 1.781 | 1.3e-4 |

- The train/validation gap opens after ~2 epochs, and validation starts rising after ~6 epochs.
  Early stopping keeps the step-3,500 checkpoint.
- This fits Muennighoff et al.: repeated epochs keep helping, with diminishing returns, and
  beyond ~4–6 epochs the added value is small.

**Muon** (train loss is a 50-step mean):

| Step | Epoch | Train | Val | LR |
|---|---|---|---|---|
| 1,000 | 1.7 | 2.168 | 2.159 | 4e-3 (constant) |
| 3,000 | 5.1 | 2.026 | 2.079 | 4e-3 |
| 4,500 | 7.6 | 2.063 | 2.066 | 4e-3 |
| 5,000 | 8.4 | 1.915 | 1.915 | cooldown |
| 5,500 | 9.3 | 1.597 | 1.716 | cooldown |
| 5,920 | 10.0 | 1.483 | **1.587** | 0 |

**Result: Muon + WSD beats AdamW + cosine by 0.158 nats** (0.5495 vs 0.6046 bpb on the full
validation set, 9% lower).

The claim needs to be stated precisely:
- **What happens during the run:** for 80% of the run Muon at a constant lr 4e-3 with wd 1.0 sits
  on a high, flat plateau. Train ≈ validation ≈ 2.07, so it is heavily regularised and does not
  overfit. Nearly all of its gain (2.07 → 1.59) comes in the cooldown.
- **AdamW is ahead at matched steps until the cooldown.** At step 3,000 AdamW is at 1.78 vs 2.08.
  So this is not "Muon learns faster per token" as in the short proxy. The short proxies (stage 1)
  overstated Muon's per-token advantage, as Wen et al. warn.
- **AdamW overfits by epoch ~6** (train 1.23, val rising). Muon's train/val gap at the end is
  0.10, against AdamW's 0.33 at its best checkpoint. So Muon got more out of 10 epochs of the
  same data.
- **The comparison is optimizer + schedule as each is usually run.** It does not isolate the
  optimizer. The clean ablation, AdamW + WSD or Muon + cosine, was not run for lack of time.
- **Muon's best checkpoint is its last step,** so a longer horizon (more epochs, or a longer
  cooldown) might help further. Untested.
- **Selected for the classifier: Muon** (lowest full validation loss).

**Known data issue:**
- 2,536 of the 45,000 training stories (5.6%) contain mojibake: UTF-8 curly quotes mis-decoded
  as cp1252, e.g. `â€œ`, `â€™`. This is present in the TinyStories source text.
- The model occasionally reproduces it at higher temperatures.
- The classifier CSV contains none, so grading is unaffected.
- It was found after the sweep had started. Fixing it mid-way would have made the final
  AdamW/Muon runs incomparable with the stages before them, so it is left as is.
- Fix for a future run: `s.encode("cp1252").decode("utf-8")`, or `ftfy.fix_text`, in `normalize()`.

## 7. Classifier

- **Model:** the pretrained Qwen3_small backbone plus a linear head on the hidden state of the
  last non-pad token.
  - Attention is causal, so the last token is the only position that has seen the whole sentence,
    and right padding cannot affect it (unit-tested).
  - Mean pooling is run as an ablation.
- **Grading label:** the CSV's `sentiment` column (positive/negative). The 6-class `emotion`
  column is reported as a secondary result.
- **Split:**
  - 31 duplicate texts removed first;
  - then a stratified 80/10/10 split (seed 0);
  - model selection on validation macro-F1; the test split is evaluated once.
- **Training:**
  - class-weighted cross-entropy (sentiment is 55/45; emotion is imbalanced);
  - AdamW, backbone lr 5e-5, head lr 1e-3, 3 epochs.
- **Reference rows** on the same test split:

  | Model | sentiment acc / macro-F1 | emotion acc / macro-F1 |
  |---|---|---|
  | Majority class | 0.549 / 0.354 | 0.335 / 0.084 |
  | TF-IDF (1–2-gram) + logistic regression | 0.939 / 0.938 | 0.866 / 0.841 |
  | Frozen backbone, linear probe | 0.722 / 0.720 | |
  | **Full fine-tune, last-token** | **0.974 / 0.974** | **0.928 / 0.904** |
  | Full fine-tune, mean-pool | 0.967 / 0.967 | |
  | Same architecture, random init, full fine-tune | 0.947 / 0.947 | |

  The random-init row isolates how much the pretraining itself contributes.

- **Backbone:** the Muon final checkpoint. Test split: 1,597 texts, so one standard error is
  about ±0.4 points.
- **Pretraining adds 2.7 points** over the same architecture trained from scratch (0.974 vs
  0.947), and the pretrained model beats TF-IDF by 3.5 points. On sentiment there are 42 errors
  in 1,597 (22 negatives called positive, 20 the other way).
- **On 6-class emotion** it beats TF-IDF by 6 points of macro-F1 (0.904 vs 0.841). The weakest
  classes are the rarest: surprise 0.85 F1, love 0.87.
- **Last-token pooling beats mean pooling** by 0.6 points, which is within ~1.5 standard errors,
  so it is a weak preference. It fits the causal-attention argument above.
- **The frozen probe is weak** (0.72). Next-token features learned on children's stories do not
  linearly separate this domain's sentiment, so fine-tuning is needed.
- **The random-init row is high** (0.947, above TF-IDF). Much of this task can be solved from
  keywords, which is also why the pretraining gain is a few points rather than tens.

## 8. Mixture-of-Experts: tested as an alternative to the dense SwiGLU

Qwen3 itself ships MoE models (Qwen3-30B-A3B, 235B-A22B: 128 experts, 8 active, no shared
experts; Qwen3 Technical Report, arXiv 2025, not peer-reviewed). An MoE variant is therefore still
a modified Qwen3. The dense model was the default for three reasons.

**Why dense was the prior:**

1. **The spec caps total parameters, not compute.**
   - An MoE's experts all count toward the 100–200M cap, while each token uses only k of them. At
     the cap, an MoE does less computation per token than a dense model of the same size.
   - The documented MoE gains come from more total parameters at matched compute: Switch
     Transformer (Fedus et al., JMLR 2022) and Krajewski et al. (ICML 2024). Under a total-parameter
     cap that advantage disappears.
   - Ludziejewski et al. (ICML 2025): at equal memory, an MoE with E ≤ 8 experts beats a
     compute-optimal dense model only when trained on about E× more tokens. We have the opposite:
     ~0.09 unique tokens per parameter.
2. **We are data-limited, and MoE adds memorisation capacity.**
   - Jelassi et al., "Mixture of Parrots" (ICLR 2025): extra experts improve memorisation more than
     reasoning, and at equal total parameters MoE generalises worse on reasoning tasks.
   - Fedus et al. (JMLR 2022) and Zoph et al., ST-MoE (arXiv 2022, not peer-reviewed): sparse
     models overfit small datasets. ST-MoE shows dense beating sparse on a 250-example task.
   - Jha et al., "Mixtures-of-Experts overfit more to repeated data" (arXiv 2026, not
     peer-reviewed): with repeated data, 80M dense models tolerate ~8 repetitions, while MoEs
     degrade from ~4 and lose to dense by ~32. More or finer experts make this worse. Dropout 0.4
     lets MoE recover.
3. **It is slower on this hardware.** PyTorch has no grouped-GEMM kernel for MPS (`grouped_mm` is
   CUDA-only), so the routed computation cannot be made sparse cheaply.

**Implementation** (`MoEFeedForward`, following the reference `standalone-qwen3-moe.ipynb`):
- **Kept from the reference:**
  - the bias-free `gate` router and per-expert SwiGLU `fc1`/`fc2`/`fc3`;
  - top-k, then softmax over the k scores;
  - every layer MoE, no shared experts;
  - the config keys `num_experts`, `num_experts_per_tok`, `moe_hidden_dim`.
- **Added for training,** since the reference is inference-only:
  - the Switch load-balancing loss E·Σ f_i·P_i, coefficient 0.01, averaged over layers;
  - optional expert dropout.
- **Router on AdamW:** the router is a classifier over experts, not a hidden-layer matrix, so it
  stays on AdamW even in Muon runs.
- **Static-shape dispatch:**
  - The reference's per-expert gather/scatter has data-dependent shapes. `torch.compile` cannot
    trace those, and the Metal code generator cannot emit `scatter`.
  - Instead, all experts run as one wide SwiGLU over the concatenated expert weights. Each
    expert's hidden units are scaled by its combine weight, which is 0 for unselected experts.
  - Output and gradients equal the reference loop (unit-tested).
  - Because every expert is computed, wall-clock does not show MoE's FLOP saving, so models are
    compared at equal tokens.

**Variants** (body = `levine`, 24 × 640):

| Preset | Experts × hidden, top-k | Total / active params | Throughput (MPS, compiled) |
|---|---|---|---|
| dense `levine` (reference) | SwiGLU 1,728 | 114.4M / 114.4M | 10.2k tok/s |
| `moe_cap` | 8 × 432, top-2 | 194.1M / 74.7M | 7.7k tok/s |
| `moe_cap_fine` | 32 × 108, top-8 (Qwen3/DeepSeekMoE-style granularity) | 194.5M / 75.1M | 6.2k tok/s |
| `moe_matched` | 8 × 216, top-2 | 114.5M / 54.8M | 11.2k tok/s |

**Pre-registered prediction:**
- At 1 epoch, the MoE models roughly match or trail dense `levine`.
- At 6 epochs, `moe_cap` overfits earlier than dense, and expert dropout 0.4 narrows the gap.
- The fine-grained variant is not better.

**Results:** *pending* (stage 7, `runs/moe/summary.csv`).

| Run | Protocol | Dense reference | MoE result |
|---|---|---|---|
| `moe_cap_1ep` | 50k × 1 epoch, wd 0.1 (shapes stage) | levine 0.6669 bpb | *pending* |
| `moe_matched_1ep` | same | levine 0.6669 bpb | *pending* |
| `moe_cap_fine_1ep` | same | levine 0.6669 bpb | *pending* |
| `moe_cap_6ep` | 20k × 6 epochs, wd 1.0 (wd stage) | levine 1.822 val loss | *pending* |
| `moe_cap_6ep_edrop0.4` | same, expert dropout 0.4 | levine 1.822 val loss | *pending* |
