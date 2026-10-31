"""Fine-tune the pretrained Qwen3_small as a sentiment (or emotion) classifier,
graded on emotions_classified.csv.

The classifier is the pretrained backbone plus a linear head on the hidden
state of the last real (non-pad) token. Because attention is causal, that is
the only position that has seen the whole sentence, and right padding after it
cannot affect it.

    uv run python classify.py --ckpt runs/final/adamw/best.pt --mode full --label sentiment
    uv run python classify.py --ckpt runs/final/adamw/best.pt --mode probe
    uv run python classify.py --ckpt runs/final/adamw/best.pt --mode random   # same model, untrained weights
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from qwen3_small.data import ROOT, get_tokenizer, normalize
from qwen3_small.model import Qwen3Model

CSV = ROOT / "artifacts" / "spec" / "emotions_classified.csv"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SPLIT_SEED = 0
MAX_LEN = 96  # longest CSV text is 88 tokens with the 8k tokenizer, so nothing is truncated


def load_splits():
    """Dedupe texts, then a stratified 80/10/10 split on `emotion` (which also stratifies `sentiment`)."""
    df = pd.read_csv(CSV)
    df["text"] = df["text"].map(normalize)
    df = df.drop_duplicates("text").reset_index(drop=True)
    train, rest = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=SPLIT_SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["emotion"], random_state=SPLIT_SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class Qwen3Classifier(nn.Module):
    def __init__(self, backbone, n_classes, pool="last"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = nn.Linear(backbone.cfg["emb_dim"], n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, lengths):
        h = self.backbone(ids, return_hidden=True)  # (b, t, d)
        if self.pool == "last":
            feats = h[torch.arange(h.size(0), device=h.device), lengths - 1]
        else:
            mask = (torch.arange(h.size(1), device=h.device)[None] < lengths[:, None]).unsqueeze(-1)
            feats = (h * mask).sum(1) / lengths[:, None]
        return self.head(feats.float())


def encode(tok, texts):
    seqs = [s[:MAX_LEN] for s in tok.encode_batch(list(texts))]
    lengths = torch.tensor([max(1, len(s)) for s in seqs])
    ids = torch.full((len(seqs), MAX_LEN), tok.pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s if s else [tok.pad_id])
    return ids, lengths


def build_classifier(ckpt_path, n_classes, mode, pool, seed=0):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt["cfg"])
    torch.manual_seed(seed)
    backbone = Qwen3Model(cfg)
    if mode != "random":
        backbone.load_state_dict(ckpt["model"])
    if mode == "probe":
        for p in backbone.parameters():
            p.requires_grad_(False)
    return Qwen3Classifier(backbone, n_classes, pool), get_tokenizer(ckpt["vocab"])


@torch.no_grad()
def predict(model, ids, lengths, bs=128):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(ids[i : i + bs].to(DEVICE), lengths[i : i + bs].to(DEVICE))
        out.append(logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["full", "probe", "random"], default="full")
    p.add_argument("--label", choices=["sentiment", "emotion"], default="sentiment")
    p.add_argument("--pool", choices=["last", "mean"], default="last")
    p.add_argument("--lr", type=float, default=5e-5, help="backbone learning rate")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    a = p.parse_args(argv)

    name = a.name or f"{a.label}_{a.mode}_{a.pool}"
    out_dir = Path("runs") / "cls" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits()
    classes = sorted(train[a.label].unique())
    to_y = lambda df: torch.tensor(df[a.label].map({c: i for i, c in enumerate(classes)}).values)
    y_tr, y_va, y_te = to_y(train), to_y(val), to_y(test)

    model, tok = build_classifier(a.ckpt, len(classes), a.mode, a.pool, a.seed)
    model.to(DEVICE)
    x_tr, l_tr = encode(tok, train.text)
    x_va, l_va = encode(tok, val.text)
    x_te, l_te = encode(tok, test.text)

    counts = np.bincount(y_tr.numpy(), minlength=len(classes))
    class_w = torch.tensor(len(y_tr) / (len(classes) * counts), dtype=torch.float32, device=DEVICE)

    groups = [{"params": model.head.parameters(), "lr": a.head_lr}]
    if a.mode != "probe":
        groups.append({"params": model.backbone.parameters(), "lr": a.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(y_tr) / a.bs)
    total = steps_per_epoch * a.epochs
    warmup = max(1, total // 10)

    print(f"{name}: {len(train)} train / {len(val)} val / {len(test)} test, classes {classes}")
    g = torch.Generator().manual_seed(a.seed)
    best_f1, best_state, step, history = -1.0, None, 0, []
    for epoch in range(a.epochs):
        model.train()
        if a.mode == "probe":
            model.backbone.eval()
        perm = torch.randperm(len(y_tr), generator=g)
        for i in range(0, len(perm), a.bs):
            idx = perm[i : i + a.bs]
            mult = (step + 1) / warmup if step < warmup else max(0.0, (total - step) / (total - warmup))
            for gr, lr in zip(opt.param_groups, base_lrs):
                gr["lr"] = lr * mult
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                logits = model(x_tr[idx].to(DEVICE), l_tr[idx].to(DEVICE))
            loss = F.cross_entropy(logits.float(), y_tr[idx].to(DEVICE), weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{total} | loss {loss.item():.4f}")
        pv = predict(model, x_va, l_va)
        f1 = f1_score(y_va, pv, average="macro")
        acc = accuracy_score(y_va, pv)
        history.append({"epoch": epoch + 1, "val_acc": acc, "val_macro_f1": f1})
        print(f"epoch {epoch + 1} | val acc {acc:.4f} | val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pt = predict(model, x_te, l_te)
    result = {
        "name": name, "mode": a.mode, "label": a.label, "pool": a.pool, "lr": a.lr, "epochs": a.epochs,
        "ckpt": a.ckpt, "classes": classes, "history": history,
        "test_acc": accuracy_score(y_te, pt), "test_macro_f1": f1_score(y_te, pt, average="macro"),
        "report": classification_report(y_te, pt, target_names=classes, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_te, pt).tolist(),
    }
    print(classification_report(y_te, pt, target_names=classes, digits=4, zero_division=0))
    print(f"TEST {name}: acc {result['test_acc']:.4f} | macro-F1 {result['test_macro_f1']:.4f}")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
