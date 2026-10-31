"""Qwen3-MoE variant: matches the reference routing, stays in the spec's parameter range."""

import pytest
import torch

import train
from qwen3_small import model as qm
from qwen3_small.config import MOE_PRESETS, make_config


def reference_moe_forward(ff, x):
    """MoEFeedForward.forward from the reference standalone-qwen3-moe.ipynb, verbatim apart from `self` -> `ff`."""
    scores = ff.gate(x)
    topk_scores, topk_indices = torch.topk(scores, ff.num_experts_per_tok, dim=-1)
    topk_probs = torch.softmax(topk_scores, dim=-1)
    batch, seq_len, _ = x.shape
    x_flat = x.reshape(batch * seq_len, -1)
    out_flat = torch.zeros(batch * seq_len, ff.emb_dim, device=x.device, dtype=x.dtype)
    topk_indices_flat = topk_indices.reshape(-1, ff.num_experts_per_tok)
    topk_probs_flat = topk_probs.reshape(-1, ff.num_experts_per_tok)
    for expert_id_tensor in torch.unique(topk_indices_flat):
        expert_id = int(expert_id_tensor.item())
        mask = topk_indices_flat == expert_id
        token_mask = mask.any(dim=-1)
        selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
        if selected_idx.numel() == 0:
            continue
        expert_input = x_flat.index_select(0, selected_idx)
        hidden = torch.nn.functional.silu(ff.fc1[expert_id](expert_input)) * ff.fc2[expert_id](expert_input)
        expert_out = ff.fc3[expert_id](hidden)
        mask_selected = mask[selected_idx]
        slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
        selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)
        out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))
    return out_flat.reshape(batch, seq_len, ff.emb_dim)


@pytest.mark.parametrize("preset", list(MOE_PRESETS))
def test_moe_matches_reference_routing(preset):
    torch.manual_seed(0)
    cfg = make_config(preset, 8192, emb_dim=64, moe_hidden_dim=16)
    ff = qm.MoEFeedForward(cfg)
    for p in ff.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(2, 7, 64, requires_grad=True)

    ours, aux, load = ff(x)
    ref = reference_moe_forward(ff, x)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)

    g = torch.randn_like(ours)
    grads_ours = torch.autograd.grad(ours, [x, *ff.parameters()], g)
    grads_ref = torch.autograd.grad(ref, [x, *ff.parameters()], g, allow_unused=True)
    for a, b in zip(grads_ours, grads_ref):
        torch.testing.assert_close(a, torch.zeros_like(a) if b is None else b, atol=1e-5, rtol=1e-5)

    assert torch.isclose(load.sum(), torch.tensor(1.0))
    assert torch.isfinite(aux) and aux.item() > 0  # E * sum(f * p) is ~1 for near-uniform routing


@pytest.mark.parametrize("preset", list(MOE_PRESETS))
def test_moe_param_count_in_spec_range(preset):
    m = qm.Qwen3Model(make_config(preset, 8192))
    b = m.param_breakdown()
    assert 100e6 <= b["total"] <= 200e6, f"{preset}: {b['total'] / 1e6:.1f}M"
    assert b["active"] < b["total"]


def test_moe_model_causal_and_returns_aux():
    torch.manual_seed(0)
    m = qm.Qwen3Model(make_config("moe_cap", 8192, n_layers=2, dropout=0.0)).eval()
    x = torch.randint(0, 8192, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 8192
    logits, aux, load = m(x, return_aux=True)
    assert torch.allclose(logits[0, :10], m(y)[0, :10], atol=1e-5)
    assert load.shape == (2, 8) and torch.isfinite(aux)


def test_router_stays_on_adamw():
    m = qm.Qwen3Model(make_config("moe_cap", 8192, n_layers=2))
    args = train.parse_args(["--name", "x", "--optim", "muon"])
    muon, adamw = train.build_optimizers(m, args)
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    assert all(id(b.ff.gate.weight) not in muon_ids for b in m.trf_blocks)
    assert all(id(b.ff.fc1[0].weight) in muon_ids for b in m.trf_blocks)


def test_moe_overfits_tiny_batch():
    torch.manual_seed(0)
    cfg = make_config("moe_cap", 512, n_layers=2, emb_dim=128, n_heads=2, n_kv_groups=1, moe_hidden_dim=64, dropout=0.0)
    m = qm.Qwen3Model(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 512, (4, 33))
    for _ in range(150):
        logits, aux, _ = m(x[:, :-1], return_aux=True)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), x[:, 1:].reshape(-1))
        opt.zero_grad()
        (loss + 0.01 * aux).backward()
        opt.step()
    assert loss.item() < 0.1
