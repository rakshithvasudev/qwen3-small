"""Spec: generation for 'Once upon a time' at temperatures 0.5/0.8/1.0/1.2, max 30 tokens, in the spec's layout."""

import re

from generate import SPEC_TEMPERATURES, format_generations
from qwen3_small.config import make_config
from qwen3_small.data import get_tokenizer
from qwen3_small.model import Qwen3Model


def test_generation_format_matches_spec():
    tok = get_tokenizer(8192)
    m = Qwen3Model(make_config("mobilellm", tok.vocab_size, n_layers=2))
    out = format_generations(m, tok, ["Once upon a time"], SPEC_TEMPERATURES)
    lines = out.splitlines()
    assert lines[0] == "📝 Testing prompt: 'Once upon a time'"
    assert lines[1] == "-" * 40
    for t in SPEC_TEMPERATURES:
        assert any(l.startswith(f"Temperature {t}: 'once upon a time") for l in lines), t
    stops = [l for l in lines if l.strip().startswith("Stopped")]
    assert len(stops) == 4
    for s in stops:
        n = re.search(r"(\d+) tokens", s)
        assert n and int(n.group(1)) <= 30
