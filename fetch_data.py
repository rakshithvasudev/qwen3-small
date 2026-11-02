"""Download the first 105k stories of roneneldan/TinyStories (train split) to data/ts105k.jsonl.

    uv run --with datasets python fetch_data.py

The first 50k unique stories are the pool every spec-range run draws from; the next 50k are only
used by the 100k-story run (see qwen3_small/data.py).
"""

import json

from datasets import load_dataset

from qwen3_small.data import STORIES_FILE

N = 105_000

if __name__ == "__main__":
    STORIES_FILE.parent.mkdir(exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    with open(STORIES_FILE, "w") as f:
        for i, ex in enumerate(ds):
            if i >= N:
                break
            f.write(json.dumps(ex["text"]) + "\n")
    print(f"wrote {N} stories to {STORIES_FILE}")
