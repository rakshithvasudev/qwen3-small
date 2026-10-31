"""Spec: 'Train a sentiment classifier using the model you just created and grade it against emotions_classified.csv'."""

import torch

from classify import build_classifier, encode, load_splits
from qwen3_small.config import make_config
from qwen3_small.model import Qwen3Model


def test_split_dedup_stratified_no_leakage():
    train, val, test = load_splits()
    texts = [set(d.text) for d in (train, val, test)]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])
    n = len(train) + len(val) + len(test)
    assert n == 16000 - 31  # 31 duplicate texts removed
    assert abs(len(train) / n - 0.8) < 0.002
    for d in (val, test):
        for label in ("sentiment", "emotion"):
            a = train[label].value_counts(normalize=True)
            b = d[label].value_counts(normalize=True)
            assert (a - b).abs().max() < 0.01


def test_backbone_is_our_pretrained_checkpoint(tmp_path):
    cfg = make_config("mobilellm", 8192, n_layers=2)
    m = Qwen3Model(cfg)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg, "vocab": 8192}, ckpt)
    clf, _ = build_classifier(ckpt, 2, "full", "last")
    for k, v in m.state_dict().items():
        assert torch.equal(v, clf.backbone.state_dict()[k]), k
    rnd, _ = build_classifier(ckpt, 2, "random", "last", seed=123)
    assert not torch.equal(rnd.backbone.tok_emb.weight, m.tok_emb.weight)


def test_last_token_pooling_ignores_padding(tmp_path):
    cfg = make_config("mobilellm", 8192, n_layers=2, dropout=0.0)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": Qwen3Model(cfg).state_dict(), "cfg": cfg, "vocab": 8192}, ckpt)
    clf, tok = build_classifier(ckpt, 2, "full", "last")
    clf.eval()
    ids, lengths = encode(tok, ["i feel happy today"])
    short = ids[:, : lengths[0]]
    with torch.no_grad():
        assert torch.allclose(clf(ids, lengths), clf(short, lengths), atol=1e-5)
