"""Spec: 'Modify the reference model and create a Qwen3_small version with ~100M-200M parameters'."""

import pytest
import torch

from qwen3_small import model as qm
from qwen3_small.config import PRESETS, QWEN_CONFIG_06_B, make_config


def test_reference_structure_kept():
    for name in ["Qwen3Model", "TransformerBlock", "GroupedQueryAttention", "FeedForward", "RMSNorm",
                 "compute_rope_params", "apply_rope"]:
        assert hasattr(qm, name), name


def test_config_keys_match_reference():
    cfg = make_config("mobilellm", 8192)
    assert set(QWEN_CONFIG_06_B) <= set(cfg)


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("vocab", [8192, 16384])
def test_param_count_in_spec_range(preset, vocab):
    n = qm.Qwen3Model(make_config(preset, vocab)).num_params()
    assert 100e6 <= n <= 200e6, f"{preset}/{vocab}: {n / 1e6:.1f}M"


def test_embeddings_tied():
    m = qm.Qwen3Model(make_config("mobilellm", 8192, n_layers=2))
    assert m.out_head.weight is m.tok_emb.weight


def test_causal():
    torch.manual_seed(0)
    m = qm.Qwen3Model(make_config("mobilellm", 8192, n_layers=2, dropout=0.0)).eval()
    x = torch.randint(0, 8192, (1, 32))
    y = x.clone()
    y[0, 20:] = torch.randint(0, 8192, (12,))
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-5)
    assert not torch.allclose(a[0, 20:], b[0, 20:], atol=1e-5)


def test_sdpa_attention_matches_reference_math():
    """Fused GQA attention equals the reference's repeat_interleave + masked softmax."""
    torch.manual_seed(0)
    att = qm.GroupedQueryAttention(d_in=64, num_heads=4, num_kv_groups=2, head_dim=16, qk_norm=True).eval()
    cos, sin = qm.compute_rope_params(16, context_length=10)
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        out = att(x, cos, sin)
        b, t, _ = x.shape
        q = att.q_norm(att.W_query(x).view(b, t, 4, 16).transpose(1, 2))
        k = att.k_norm(att.W_key(x).view(b, t, 2, 16).transpose(1, 2))
        v = att.W_value(x).view(b, t, 2, 16).transpose(1, 2)
        q, k = qm.apply_rope(q, cos, sin), qm.apply_rope(k, cos, sin)
        k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        w = torch.softmax((q @ k.transpose(2, 3)).masked_fill(mask, -torch.inf) / 16**0.5, dim=-1)
        ref = att.out_proj((w @ v).transpose(1, 2).reshape(b, t, 64))
    assert torch.allclose(out, ref, atol=1e-5)


def test_overfits_tiny_batch():
    """The training step is correct if a small model can memorise 4 sequences."""
    torch.manual_seed(0)
    m = qm.Qwen3Model(make_config("mobilellm", 8192, n_layers=2, dropout=0.0))
    x = torch.randint(0, 8192, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, 8192), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
