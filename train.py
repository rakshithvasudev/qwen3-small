"""Pretrain Qwen3_small on a TinyStories subset.

Prints the training loss at every optimizer step and the validation loss at
every evaluation, and writes both to runs/<name>/metrics.jsonl.

Example (a ~3-minute run from scratch, 4 validation evals):
    uv run python train.py --name demo --config levine --stories 10000 --epochs 1 \
        --optim muon --lr 4e-3 --sched wsd --warmup-frac 0.1 --wd 1.0 --eval-every 20 --max-steps 60
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generate import SPEC_PROMPTS, SPEC_TEMPERATURES, format_generations
from qwen3_small.config import make_config
from qwen3_small.data import POOL_SIZE, bits_per_byte, get_dataset, get_tokenizer, make_windows
from qwen3_small.model import Qwen3Model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config", default="mobilellm")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--stories", type=int, default=50_000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--optim", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--sched", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--micro-bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-windows", type=int, default=256)
    p.add_argument("--patience", type=int, default=0, help="stop after N evals without improvement (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--save", action="store_true", help="save best checkpoint to runs/<name>/best.pt")
    p.add_argument("--max-steps", type=int, default=0, help="cap on optimizer steps (0 = from epochs)")
    p.add_argument("--pause-every", type=float, default=0,
                   help="pause for --pause-for minutes after every N minutes of training (0 = never)")
    p.add_argument("--pause-for", type=float, default=10)
    p.add_argument("--expert-dropout", type=float, default=0.0, help="MoE presets: dropout inside experts")
    p.add_argument("--moe-aux", type=float, default=0.01,
                   help="load-balancing loss coefficient for MoE presets (Switch Transformer default)")
    p.add_argument("--quiet", action="store_true", help="print train loss every 10 steps instead of every step")
    return p.parse_args(argv)


def lr_multiplier(step, total, warmup, sched):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    if sched == "cosine":
        # cosine decay to 1% of peak
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    # WSD: constant, then 1 - sqrt cooldown to 0 over the last 20% of steps
    cooldown_start = 0.8
    if progress < cooldown_start:
        return 1.0
    return 1.0 - math.sqrt((progress - cooldown_start) / (1 - cooldown_start))


def build_optimizers(model, args):
    if args.optim == "adamw":
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, args.beta2), eps=1e-8,
        )
        return [opt]
    # Muon for the 2D weight matrices inside the transformer blocks; AdamW for the
    # tied embedding/head and all norm gains (Jordan 2024; Liu et al. 2025). An MoE
    # router (`ff.gate`, experts x d) is a classifier over experts, not a hidden-layer
    # matrix, so it also stays on AdamW.
    muon_params = [p for n, p in model.trf_blocks.named_parameters() if p.dim() == 2 and not n.endswith("ff.gate.weight")]
    muon_ids = {id(p) for p in muon_params}
    rest = [p for p in model.parameters() if id(p) not in muon_ids]
    rest_decay = [p for p in rest if p.dim() >= 2]
    rest_no_decay = [p for p in rest if p.dim() < 2]
    muon = torch.optim.Muon(
        muon_params, lr=args.lr, weight_decay=args.wd, momentum=0.95, nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        [{"params": rest_decay, "weight_decay": args.wd}, {"params": rest_no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, args.beta2), eps=1e-8,
    )
    return [muon, adamw]


@torch.no_grad()
def evaluate(model, windows, bs):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(windows), bs):
        batch = torch.from_numpy(windows[i : i + bs]).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), reduction="sum")
        total += loss.item()
        n += batch[:, 1:].numel()
    model.train()
    return total / n


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = get_tokenizer(args.vocab)
    train_tokens, val_tokens, meta = get_dataset(args.stories, args.vocab)
    extra = {"expert_dropout": args.expert_dropout} if args.expert_dropout else {}
    cfg = make_config(args.config, tok.vocab_size, dropout=args.dropout, **extra)
    seq_len = cfg["context_length"]
    train_w = make_windows(train_tokens, seq_len)
    val_w = make_windows(val_tokens, seq_len)
    eval_w = val_w[: args.eval_windows]

    model = Qwen3Model(cfg).to(DEVICE)
    is_moe = cfg.get("num_experts", 0) > 0
    fwd = torch.compile(model) if args.compile else model
    optimizers = build_optimizers(model, args)

    tokens_per_step = args.micro_bs * args.accum * seq_len
    steps_per_epoch = len(train_w) // (args.micro_bs * args.accum)
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * args.warmup_frac))
    params = model.param_breakdown()
    # Early stopping only counts evals while the LR is decaying. Under WSD the constant phase
    # sits on a plateau and the loss drops in the cooldown, so stopping before it is premature.
    patience_from = warmup + int(0.8 * (total_steps - warmup)) if args.sched == "wsd" else 0

    header = {
        "args": vars(args), "cfg": {k: str(v) if k == "dtype" else v for k, v in cfg.items()},
        "params": params, "data": meta, "tokens_per_step": tokens_per_step,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "device": DEVICE,
    }
    (out_dir / "config.json").write_text(json.dumps(header, indent=2))
    print(f"model {args.config}: {params['total'] / 1e6:.1f}M params "
          f"({params['non_embedding'] / 1e6:.1f}M non-embedding, {params['active'] / 1e6:.1f}M active per token), "
          f"vocab {tok.vocab_size}")
    print(f"data: {meta['n_train_stories']} train / {meta['n_val_stories']} val stories, "
          f"{meta['train_tokens']:,} train tokens, {len(train_w)} train windows of {seq_len}")
    print(f"optim {args.optim} lr {args.lr} wd {args.wd} sched {args.sched}; "
          f"{tokens_per_step:,} tokens/step, {steps_per_epoch} steps/epoch, {total_steps} steps total")

    log = open(out_dir / "metrics.jsonl", "w")

    def record(**row):
        log.write(json.dumps(row) + "\n")
        log.flush()

    order = rng.permutation(len(train_w))
    cursor = 0
    best_val, bad_evals, best_step = float("inf"), 0, 0
    t0 = time.time()
    tokens_seen = 0
    model.train()

    def run_eval(step):
        nonlocal best_val, bad_evals, best_step
        v = evaluate(fwd, eval_w, args.micro_bs)
        bpb = bits_per_byte(v, meta)
        improved = v < best_val
        if improved:
            best_val, bad_evals, best_step = v, 0, step
            if args.save:
                torch.save({"model": model.state_dict(), "cfg": cfg, "vocab": args.vocab, "step": step},
                           out_dir / "best.pt")
        elif step >= patience_from:
            bad_evals += 1
        print(f"step {step:5d} | val loss {v:.4f} | val bpb {bpb:.4f}{' *' if improved else ''}")
        record(step=step, val_loss=v, val_bpb=bpb, tokens=tokens_seen, elapsed=time.time() - t0)
        return args.patience and bad_evals >= args.patience

    run_eval(0)
    stopped_early = False
    last_pause = time.time()
    for step in range(1, total_steps + 1):
        mult = lr_multiplier(step - 1, total_steps, warmup, args.sched)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = args.lr * mult

        loss_acc, aux_acc, load = 0.0, 0.0, None
        for _ in range(args.accum):
            if cursor + args.micro_bs > len(order):
                order, cursor = rng.permutation(len(train_w)), 0
            idx = order[cursor : cursor + args.micro_bs]
            cursor += args.micro_bs
            batch = torch.from_numpy(train_w[idx]).to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                if is_moe:
                    logits, aux, load = fwd(batch[:, :-1], return_aux=True)
                else:
                    logits = fwd(batch[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            total = loss + args.moe_aux * aux / cfg["n_layers"] if is_moe else loss
            (total / args.accum).backward()
            loss_acc += loss.detach()
            if is_moe:
                aux_acc += aux.detach() / cfg["n_layers"]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        train_loss = (loss_acc / args.accum).item()
        if not math.isfinite(train_loss):
            print(f"step {step:5d} | train loss {train_loss} -> diverged, stopping")
            record(step=step, train_loss=None, diverged=True)
            break
        elapsed = time.time() - t0
        tok_s = tokens_seen / elapsed
        if not args.quiet or step % 10 == 0 or step == 1:
            moe_txt = f" | moe aux {(aux_acc / args.accum).item():.3f}" if is_moe else ""
            print(f"step {step:5d} | train loss {train_loss:.4f} | lr {args.lr * mult:.2e} | "
                  f"grad {grad_norm.item():.2f} | {tok_s:,.0f} tok/s | epoch {tokens_seen / meta['train_tokens']:.2f}{moe_txt}")
        row = dict(step=step, train_loss=train_loss, lr=args.lr * mult, grad_norm=grad_norm.item(),
                   tokens=tokens_seen, tok_s=tok_s, elapsed=elapsed)
        if is_moe:
            # aux = 1.0 is perfectly balanced; max_load = busiest expert's share x num_experts (1.0 = balanced)
            row.update(moe_aux=(aux_acc / args.accum).item(),
                       moe_max_load=(load.max(dim=-1).values.mean() * cfg["num_experts"]).item())
        record(**row)

        if step % args.eval_every == 0 or step == total_steps:
            if run_eval(step):
                print(f"early stop at step {step}: no val improvement for {args.patience} evals")
                stopped_early = True
                break

        # Cooling break for long runs; pause time is excluded from elapsed/throughput.
        if args.pause_every and time.time() - last_pause > args.pause_every * 60 and step < total_steps:
            print(f"step {step:5d} | pausing {args.pause_for:g} min to cool down", flush=True)
            time.sleep(args.pause_for * 60)
            t0 += args.pause_for * 60
            last_pause = time.time()

    elapsed = time.time() - t0
    summary = {
        "name": args.name, "config": args.config, "vocab": args.vocab, "optim": args.optim, "lr": args.lr,
        "wd": args.wd, "sched": args.sched, "stories": args.stories, "params_M": params["total"] / 1e6,
        "active_params_M": params["active"] / 1e6,
        "best_val_loss": best_val, "best_val_bpb": bits_per_byte(best_val, meta), "best_step": best_step,
        "steps": step, "stopped_early": stopped_early, "tokens_seen": tokens_seen,
        "elapsed_min": elapsed / 60, "tok_s": tokens_seen / elapsed,
    }

    if args.save and (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=DEVICE)["model"])
        full_val = evaluate(model, val_w, args.micro_bs)
        summary.update(full_val_loss=full_val, full_val_bpb=bits_per_byte(full_val, meta))
        print(f"best checkpoint (step {best_step}) on full val set: loss {full_val:.4f}, "
              f"bpb {bits_per_byte(full_val, meta):.4f}")
        if args.stories > POOL_SIZE:
            # Also score on the 50k run's val set (a subset of ours) so the two runs compare directly.
            _, val50, meta50 = get_dataset(POOL_SIZE, args.vocab)
            v = evaluate(model, make_windows(val50, seq_len), args.micro_bs)
            summary.update(val50k_loss=v, val50k_bpb=bits_per_byte(v, meta50))
            print(f"best checkpoint on the 50k-run val set: loss {v:.4f}, bpb {bits_per_byte(v, meta50):.4f}")
    samples = format_generations(model, tok, SPEC_PROMPTS[:1], SPEC_TEMPERATURES, seed=args.seed)
    (out_dir / "samples.txt").write_text(samples)
    print(samples)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
