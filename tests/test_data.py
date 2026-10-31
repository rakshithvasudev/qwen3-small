"""Spec: 'Create or use a tokenizer' and 'subset of Tiny Stories ... train/test split of 90%/10% or 80%/20%'."""

import pandas as pd
import pytest

from qwen3_small.data import ROOT, get_tokenizer, load_split, normalize

CSV = ROOT / "artifacts" / "spec" / "emotions_classified.csv"


@pytest.mark.parametrize("n", [10_000, 20_000, 50_000])
def test_split_is_90_10_without_overlap(n):
    train, val = load_split(n)
    assert len(train) + len(val) == n
    assert abs(len(train) / n - 0.9) < 1e-3
    assert not set(train) & set(val)


def test_smaller_runs_are_subsets_and_val_never_in_any_train():
    train50, val50 = load_split(50_000)
    train20, val20 = load_split(20_000)
    assert set(train20) <= set(train50) and set(val20) <= set(val50)
    assert not set(val50) & set(train50)


@pytest.mark.parametrize("vocab", [8192, 16384])
def test_tokenizer_roundtrip_no_unknowns(vocab):
    tok = get_tokenizer(vocab)
    assert tok.vocab_size == vocab
    assert tok.tok.token_to_id("<unk>") is None
    _, val = load_split(10_000)
    texts = val[:200] + pd.read_csv(CSV).text.head(500).map(normalize).tolist()
    for t in texts:
        assert tok.decode(tok.encode(t)) == t
