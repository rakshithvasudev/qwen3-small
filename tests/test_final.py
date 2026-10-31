"""Checks on the final pretraining runs (skipped until runs/final/* exist)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FINAL_RUNS = sorted((ROOT / "runs" / "final").glob("*/summary.json"))


@pytest.mark.skipif(not FINAL_RUNS, reason="final runs not finished")
@pytest.mark.parametrize("summary_path", FINAL_RUNS, ids=lambda p: p.parent.name)
def test_final_run_meets_spec(summary_path):
    run = summary_path.parent
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(l) for l in open(run / "metrics.jsonl")]
    train = [r for r in rows if "train_loss" in r]
    val = [r for r in rows if "val_loss" in r]

    assert 100e6 <= cfg["params"]["total"] <= 200e6
    assert 10_000 <= cfg["args"]["stories"] <= 50_000
    assert abs(cfg["data"]["n_train_stories"] / cfg["args"]["stories"] - 0.9) < 1e-3
    # loss is logged at every optimizer step, and validated at least twice
    assert [r["step"] for r in train] == list(range(1, len(train) + 1))
    assert len(val) >= 2
    assert min(r["val_loss"] for r in val[1:]) < val[0]["val_loss"]
    assert (run / "best.pt").exists()
    samples = (run / "samples.txt").read_text()
    assert all(f"Temperature {t}:" in samples for t in (0.5, 0.8, 1.0, 1.2))
