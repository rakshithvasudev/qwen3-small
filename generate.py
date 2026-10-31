"""Text generation in the output format of the project spec.

    uv run python generate.py --ckpt runs/final/muon/best.pt
"""

import argparse

import torch

from qwen3_small.data import get_tokenizer, normalize
from qwen3_small.model import Qwen3Model

SPEC_PROMPTS = ["Once upon a time", "The little dog", "One day, Lily found a"]
SPEC_TEMPERATURES = [0.5, 0.8, 1.0, 1.2]
MAX_NEW_TOKENS = 30


@torch.no_grad()
def generate(model, tok, prompt, temperature, max_new_tokens=MAX_NEW_TOKENS, generator=None):
    """Temperature sampling. Returns (text, n_new_tokens, stop_reason)."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tok.encode(normalize(prompt))], device=device)
    new = []
    stop = "max"
    for _ in range(max_new_tokens):
        ctx = ids[:, -model.cfg["context_length"]:]
        logits = model(ctx)[:, -1].float()
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(device)
        if nxt.item() == tok.eos_id:
            stop = "eos"
            break
        new.append(nxt.item())
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0].tolist()), len(new), stop


def format_generations(model, tok, prompts, temperatures, seed=0, max_new_tokens=MAX_NEW_TOKENS):
    g = torch.Generator().manual_seed(seed)
    lines = []
    for prompt in prompts:
        lines.append(f"📝 Testing prompt: '{prompt}'")
        lines.append("-" * 40)
        for t in temperatures:
            text, n, stop = generate(model, tok, prompt, t, max_new_tokens, generator=g)
            if stop == "max":
                lines.append(f"  Stopped after maximum {max_new_tokens} tokens")
            else:
                lines.append(f"  Stopped at end of text after {n} tokens")
            lines.append(f"Temperature {t}: '{text}'")
            lines.append("")
    return "\n".join(lines)


def load_model(ckpt_path, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Qwen3Model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, get_tokenizer(ckpt["vocab"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", action="append")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    model, tok = load_model(a.ckpt)
    print(format_generations(model, tok, a.prompt or SPEC_PROMPTS, SPEC_TEMPERATURES, seed=a.seed))
