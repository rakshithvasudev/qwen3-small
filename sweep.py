"""Run one experiment stage (a grid of train.py runs) sequentially and summarise it.

    uv run python sweep.py throughput
    uv run python sweep.py lr
    uv run python sweep.py vocab --optim muon --lr 4e-3
    uv run python sweep.py shapes --lr 2e-3 --vocab 8192
    uv run python sweep.py wd --lr 2e-3 --vocab 8192 --config mobilellm
    uv run python sweep.py final --lr 2e-3 --muon-lr 2e-3 --vocab 8192 --config mobilellm --wd 0.5 --epochs 10

Each run's stdout goes to runs/<stage>/<run>/stdout.log; the stage table is
written to runs/<stage>/summary.csv.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from qwen3_small.config import PRESETS

COMMON = ["--compile", "--quiet"]


def grid(stage, a):
    """Returns a list of (run_name, train.py args)."""
    if stage == "throughput":
        return [(c, ["--config", c, "--vocab", str(a.vocab), "--stories", "10000", "--max-steps", "30",
                     "--eval-every", "1000", "--eval-windows", "16"]) for c in PRESETS]
    if stage == "lr":
        base = ["--config", a.config, "--vocab", str(a.vocab), "--stories", "20000", "--epochs", "1",
                "--sched", "wsd", "--wd", "0.1", "--eval-every", "50"]
        wu = [] if a.warmup_frac is None else ["--warmup-frac", str(a.warmup_frac)]
        tag = "" if a.warmup_frac is None else f"_wu{a.warmup_frac}"
        runs = [(f"adamw_lr{lr}{tag}", base + ["--optim", "adamw", "--lr", str(lr)] + wu) for lr in a.lrs]
        runs += [(f"muon_lr{lr}{tag}", base + ["--optim", "muon", "--lr", str(lr)] + wu) for lr in a.lrs]
        return runs
    # Stages 2-4 use one optimizer setup (--optim/--lr/--sched) and at least 10% warmup:
    # the LR stage showed 5% of a ~240-step run (11 steps) left AdamW stuck on the unigram plateau.
    opt = ["--optim", a.optim, "--lr", str(a.lr), "--sched", a.sched, "--warmup-frac", str(a.warmup_frac or 0.1)]
    if stage == "vocab":
        return [(f"vocab{v}", ["--config", a.config, "--vocab", str(v), "--stories", "50000", "--epochs", "1",
                               "--wd", "0.1"] + opt) for v in (8192, 16384)]
    if stage == "shapes":
        return [(c, ["--config", c, "--vocab", str(a.vocab), "--stories", "50000", "--epochs", "1",
                     "--wd", "0.1"] + opt) for c in PRESETS]
    if stage == "wd":
        return [(f"wd{wd}", ["--config", a.config, "--vocab", str(a.vocab), "--stories", "20000",
                             "--epochs", "6", "--wd", str(wd)] + opt) for wd in a.wds]
    if stage == "moe":
        # Qwen3-MoE variants of levine. Two protocols, each with an existing dense reference:
        #   1 epoch of 50k stories = the shapes stage (dense levine 0.6669 bpb);
        #   6 epochs of 20k stories, wd 1.0 = the wd stage (dense levine 1.822), where repeated
        #   data is expected to hurt MoE more; expert dropout 0.4 is the proposed remedy.
        one = ["--vocab", str(a.vocab), "--stories", "50000", "--epochs", "1", "--wd", "0.1"] + opt
        six = ["--vocab", str(a.vocab), "--stories", "20000", "--epochs", "6", "--wd", "1.0",
               "--pause-every", "45"] + opt
        return [("moe_cap_1ep", ["--config", "moe_cap"] + one),
                ("moe_matched_1ep", ["--config", "moe_matched"] + one),
                ("moe_cap_6ep", ["--config", "moe_cap"] + six),
                ("moe_cap_6ep_edrop0.4", ["--config", "moe_cap", "--expert-dropout", "0.4"] + six),
                ("moe_cap_fine_1ep", ["--config", "moe_cap_fine"] + one)]
    if stage == "final":
        base = ["--config", a.config, "--vocab", str(a.vocab), "--stories", "50000", "--epochs", str(a.epochs),
                "--wd", str(a.wd), "--save", "--patience", str(a.patience), "--pause-every", "45"]
        return [("adamw", base + ["--optim", "adamw", "--lr", str(a.lr), "--sched", "cosine"]),
                ("muon", base + ["--optim", "muon", "--lr", str(a.muon_lr or a.lr), "--sched", "wsd"])]
    raise ValueError(stage)


def steady_tok_s(metrics_path, skip=10):
    rows = [json.loads(l) for l in open(metrics_path)]
    steps = [r for r in rows if "train_loss" in r and r.get("train_loss") is not None]
    if len(steps) <= skip:
        return None
    a, b = steps[skip - 1], steps[-1]
    return (b["tokens"] - a["tokens"]) / (b["elapsed"] - a["elapsed"])


def thermal_state():
    """macOS thermal report (CPU speed limit 100 = no throttling)."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10).stdout
        return " ".join(l.strip() for l in out.splitlines() if "=" in l) or "no thermal warnings"
    except Exception:
        return "thermal state unavailable"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--config", default="mobilellm")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--muon-lr", type=float, default=None)
    p.add_argument("--optim", choices=["adamw", "muon"], default="adamw", help="vocab/shapes/wd stages")
    p.add_argument("--sched", choices=["cosine", "wsd"], default="wsd", help="vocab/shapes/wd stages")
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 4e-3])
    p.add_argument("--wds", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--warmup-frac", type=float, default=None, help="lr stage: override warmup fraction")
    p.add_argument("--only", nargs="+", help="run only these run names")
    p.add_argument("--cooldown", type=int, default=600, help="seconds to idle between runs to let the machine cool")
    a = p.parse_args()

    stage_dir = Path("runs") / a.stage
    rows = []
    ran_previous = False
    for name, args in grid(a.stage, a):
        if a.only and name not in a.only:
            continue
        run_name = f"{a.stage}/{name}"
        out = Path("runs") / run_name
        if (out / "summary.json").exists():
            print(f"skip {run_name} (done)")
        else:
            if ran_previous and a.cooldown:
                print(f"cooling down {a.cooldown // 60} min before next run | {thermal_state()}", flush=True)
                time.sleep(a.cooldown)
            ran_previous = True
            out.mkdir(parents=True, exist_ok=True)
            print(f"run {run_name}: {' '.join(args)}", flush=True)
            with open(out / "stdout.log", "w") as log:
                subprocess.run([sys.executable, "-u", "train.py", "--name", run_name, *args, *COMMON],
                               stdout=log, stderr=subprocess.STDOUT)
        if (out / "summary.json").exists():
            s = json.loads((out / "summary.json").read_text())
            s["steady_tok_s"] = steady_tok_s(out / "metrics.jsonl")
            rows.append(s)
            print(f"  -> best val loss {s['best_val_loss']:.4f} | bpb {s['best_val_bpb']:.4f} | "
                  f"{s['elapsed_min']:.1f} min | steady {s['steady_tok_s'] or 0:,.0f} tok/s", flush=True)
        else:
            print(f"  -> FAILED, see {out / 'stdout.log'}", flush=True)
    # The stage table covers every finished run in the stage, including earlier invocations.
    rows = []
    for f in sorted(stage_dir.glob("*/summary.json")):
        s = json.loads(f.read_text())
        s["steady_tok_s"] = steady_tok_s(f.parent / "metrics.jsonl")
        s["warmup_frac"] = json.loads((f.parent / "config.json").read_text())["args"]["warmup_frac"]
        rows.append(s)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(stage_dir / "summary.csv", index=False)
        cols = ["name", "params_M", "active_params_M", "optim", "lr", "wd", "vocab", "best_val_loss", "best_val_bpb", "best_step",
                "warmup_frac", "steps", "elapsed_min", "steady_tok_s"]
        print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
