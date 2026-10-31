"""Qwen3_small.

Adapted from the reference Qwen3 implementation
(rasbt/LLMs-from-scratch, ch05/11_qwen3/standalone-qwen3.ipynb). Class names,
module layout and parameter names are kept so the two can be compared line by
line. Changes for from-scratch training at ~100-200M parameters:

  * GroupedQueryAttention uses F.scaled_dot_product_attention (fused, causal,
    native GQA) instead of materialising the score matrix and repeating K/V.
  * Residual dropout after attention and feed-forward (multi-epoch training on
    a small dataset).
  * Input embedding and output head are tied in __init__ (the reference only
    ties them when loading a pretrained checkpoint).
  * GPT-2-style init: N(0, 0.02), with residual output projections scaled by
    1/sqrt(2 * n_layers). The reference relies on PyTorch defaults, where
    nn.Embedding is N(0, 1) and gives huge initial logits once tied.
  * forward(..., return_hidden=True) exposes final hidden states for the
    classification head.
  * MoEFeedForward follows the reference standalone-qwen3-moe.ipynb (same
    `gate` router, per-expert SwiGLU fc1/fc2/fc3, top-k then softmax over the
    k scores, every layer MoE, no shared experts). Two training additions: a
    load-balancing auxiliary loss (the reference is inference-only), and
    static-shape dispatch (see MoEFeedForward.forward).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        self.fc1 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_hidden_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc2 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_hidden_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc3 = nn.ModuleList([nn.Linear(cfg["moe_hidden_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        # Dropout on expert hidden activations ("expert dropout", Fedus et al. 2022, 0.4 when fine-tuning)
        self.expert_drop = nn.Dropout(cfg.get("expert_dropout", 0.0))

    def forward(self, x):
        """Returns (output, aux_loss, load).

        Routing is the reference's: top-k gate scores, softmax over those k. The reference
        gathers each expert's tokens with data-dependent shapes (unique/nonzero/index_add_),
        which torch.compile cannot trace into one graph and which is slow on MPS. Here every
        expert runs on every token, written as one wide SwiGLU over the concatenated experts
        (width num_experts * moe_hidden_dim) whose hidden units are scaled by their expert's
        combine weight, which is 0 for unselected experts. The output and gradients are identical
        to the reference (unit-tested). The weights stay per expert, as in the reference, so Muon
        orthogonalises each expert separately. The cost is computing all experts, so wall-clock
        does not show MoE's FLOP saving; comparisons are made at equal tokens.

        aux_loss is the Switch Transformer load-balancing loss, E * sum_i f_i * P_i, where f_i is
        the fraction of routed slots sent to expert i and P_i its mean router probability; it is
        1.0 when routing is uniform. load is f (detached), for monitoring.
        """
        b, t, _ = x.shape
        scores = self.gate(x)  # (b, t, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        # (b, t, k, E) one-hot of the chosen experts, built by comparison rather than scatter,
        # which the MPS backend of torch.compile cannot generate. Combine weights stay fp32.
        onehot = (topk_indices.unsqueeze(-1) == torch.arange(self.num_experts, device=x.device)).to(topk_probs.dtype)
        combine = (onehot * topk_probs.unsqueeze(-1)).sum(dim=-2)  # (b, t, E)

        w1 = torch.cat([fc.weight for fc in self.fc1])  # (E*h, d)
        w2 = torch.cat([fc.weight for fc in self.fc2])  # (E*h, d)
        w3 = torch.cat([fc.weight for fc in self.fc3], dim=1)  # (d, E*h)
        hidden = F.silu(F.linear(x, w1)) * F.linear(x, w2)  # (b, t, E*h)
        hidden = hidden.view(b, t, self.num_experts, -1) * combine.unsqueeze(-1).to(hidden.dtype)
        out = F.linear(self.expert_drop(hidden.view(b, t, -1)), w3)

        f = onehot.sum(dim=-2).mean(dim=(0, 1)) / self.num_experts_per_tok
        p = torch.softmax(scores.float(), dim=-1).mean(dim=(0, 1))
        aux_loss = self.num_experts * (f * p).sum()
        return out, aux_loss, f.detach()


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (context_length, head_dim // 2)
    angles = torch.cat([angles, angles], dim=1)  # (context_length, head_dim)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Fused causal attention; enable_gqa shares each K/V head across its query group
        # (equivalent to the reference's repeat_interleave + masked softmax).
        context = F.scaled_dot_product_attention(
            queries, keys, values, is_causal=True, enable_gqa=self.group_size > 1
        )
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
        )
        if cfg.get("num_experts", 0) > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))

    def forward(self, x, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin)
        x = self.drop(x) + shortcut

        shortcut = x
        x = self.norm2(x)
        moe_stats = None
        if isinstance(self.ff, MoEFeedForward):
            x, aux_loss, load = self.ff(x)
            moe_stats = (aux_loss, load)
        else:
            x = self.ff(x)
        x = self.drop(x) + shortcut

        return x, moe_stats


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg["n_layers"])
        for block in self.trf_blocks:
            nn.init.normal_(block.att.out_proj.weight, mean=0.0, std=resid_std)
            for fc3 in (block.ff.fc3 if isinstance(block.ff, MoEFeedForward) else [block.ff.fc3]):
                nn.init.normal_(fc3.weight, mean=0.0, std=resid_std)

        if cfg.get("tie_embeddings", True):
            self.out_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx, return_hidden=False, return_aux=False):
        """return_aux=True (MoE models) also returns (aux_loss summed over layers, load of shape (layers, experts))."""
        x = self.tok_emb(in_idx)
        aux, loads = [], []
        for block in self.trf_blocks:
            x, moe_stats = block(x, self.cos, self.sin)
            if moe_stats is not None:
                aux.append(moe_stats[0])
                loads.append(moe_stats[1])
        x = self.final_norm(x)
        if return_hidden:
            return x
        logits = self.out_head(x)
        if return_aux:
            if not aux:
                return logits, x.new_zeros(()), None
            return logits, torch.stack(aux).sum(), torch.stack(loads)
        return logits

    def num_params(self):
        """Unique parameters (a tied embedding/head matrix is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        emb = self.tok_emb.weight.numel()
        total = self.num_params()
        out = {"total": total, "embedding": emb, "non_embedding": total - emb, "active": total}
        if self.cfg.get("num_experts", 0) > 0:
            ff = self.trf_blocks[0].ff
            per_expert = sum(p.numel() for m in (ff.fc1[0], ff.fc2[0], ff.fc3[0]) for p in m.parameters())
            unused = self.cfg["n_layers"] * (self.cfg["num_experts"] - self.cfg["num_experts_per_tok"]) * per_expert
            out["active"] = total - unused
        return out
