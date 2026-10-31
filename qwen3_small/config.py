"""Model configurations.

Uses the same config-dict keys as the reference Qwen3 implementation
(rasbt/LLMs-from-scratch, ch05/11_qwen3/standalone-qwen3.ipynb) so the
scaled-down presets can be diffed directly against the original 0.6B config.
"""

import torch

# Original Qwen3-0.6B config, copied from the reference notebook for comparison.
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# Shared settings for every Qwen3_small preset. Changes vs 0.6B:
#   vocab_size: custom BPE trained on TinyStories (set per run, 8k or 16k)
#   context_length 40_960 -> 512: 97% of stories fit in 512 tokens
#   head_dim 128 -> 64: keeps n_heads * head_dim close to emb_dim at this width
#   rope_base 1e6 -> 1e4: 1e6 is Qwen3's long-context setting (report §3.2)
#   dtype bf16 -> fp32 weights: training uses bf16 autocast with fp32 master weights
#   dropout: added for multi-epoch training on a small dataset
_SHARED = {
    "context_length": 512,
    "head_dim": 64,
    "qk_norm": True,
    "rope_base": 10_000.0,
    "dtype": torch.float32,
    "dropout": 0.1,
    "tie_embeddings": True,
}

# Each preset tests a published claim about model shape at ~100-200M params.
# (emb_dim, n_layers, n_heads, n_kv_groups, hidden_dim)
PRESETS = {
    # MobileLLM-125M / SmolLM2-135M shape: deep and thin (Liu et al., ICML 2024)
    "mobilellm": (576, 30, 9, 3, 1536),
    # Depth-efficiency optimum for this size (Levine et al., NeurIPS 2020)
    "levine": (640, 24, 10, 5, 1728),
    # DeepNarrow, gains saturate at 32-36 layers (Tay et al., ICLR 2022)
    "deepnarrow": (512, 36, 8, 4, 1408),
    # GPT-2 / Pythia-160M depth; 12L optimal at 134M (Petty et al., NAACL 2024)
    "classic12": (768, 12, 12, 4, 3072),
    # Wide and shallow: shape matters little (Kaplan et al. 2020, arXiv)
    "wide8": (1024, 8, 16, 4, 4096),
    # MLP-heavy: MLP dim scales fastest (Alabdulmohsin et al., NeurIPS 2023)
    "mlpheavy": (640, 18, 10, 5, 3456),
}

# Qwen3-MoE variants of the `levine` body (reference: standalone-qwen3-moe.ipynb, which uses
# every layer MoE, no shared experts). The spec caps TOTAL parameters, so:
# (base preset, num_experts, num_experts_per_tok, moe_hidden_dim)
MOE_PRESETS = {
    # At the 200M cap: 8 experts x 432, top-2. 194M total, ~75M active per token.
    # Total expert width 8 x 432 = 3456 = mlpheavy's dense SwiGLU width.
    "moe_cap": ("levine", 8, 2, 432),
    # Fine-grained version of moe_cap (Qwen3 / DeepSeekMoE style): 32 x 108, top-8.
    # Same total and active parameters, 4x the routing granularity.
    "moe_cap_fine": ("levine", 32, 8, 108),
    # Matched to dense levine's TOTAL parameters: 8 x 216, top-2. 114M total, ~55M active.
    "moe_matched": ("levine", 8, 2, 216),
}


def make_config(preset, vocab_size, **overrides):
    moe = None
    if preset in MOE_PRESETS:
        base, n_exp, k, moe_hidden = MOE_PRESETS[preset]
        moe = {"num_experts": n_exp, "num_experts_per_tok": k, "moe_hidden_dim": moe_hidden}
        emb_dim, n_layers, n_heads, n_kv_groups, hidden_dim = PRESETS[base]
    else:
        emb_dim, n_layers, n_heads, n_kv_groups, hidden_dim = PRESETS[preset]
    cfg = dict(_SHARED)
    cfg.update(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_groups=n_kv_groups,
        hidden_dim=hidden_dim,
        preset=preset,
    )
    if moe:
        cfg.update(moe)
    cfg.update(overrides)
    return cfg
