"""TinyStories data: normalization, train/val split, tokenization, packing.

Split design: the first POOL_SIZE stories are shuffled once (seed 0) and cut
90/10 into a train pool and a val pool. A run with N stories takes the first
0.9N from the train pool and the first 0.1N from the val pool, so every run
size is a 90/10 split, smaller runs are subsets of larger ones, and no val
story is ever seen by the tokenizer or by training.
"""

import json
import math
import random
import re
from pathlib import Path

import numpy as np

from .tokenizer import load_tokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORIES_FILE = DATA_DIR / "ts100k.jsonl"  # first 100k stories of roneneldan/TinyStories train
POOL_SIZE = 50_000
TRAIN_FRAC = 0.9
SPLIT_SEED = 0

_APOSTROPHES = re.compile(r"[’']")


def normalize(text):
    """Lowercase and drop apostrophes ("didn't" -> "didnt") to match the classifier CSV style."""
    return _APOSTROPHES.sub("", text.lower()).strip()


def load_split(n_stories=POOL_SIZE):
    assert n_stories <= POOL_SIZE
    # TinyStories contains empty and duplicated stories; drop them before splitting so
    # no text can appear in both train and val.
    stories, seen = [], set()
    with open(STORIES_FILE) as f:
        for line in f:
            s = normalize(json.loads(line))
            if s and s not in seen:
                seen.add(s)
                stories.append(s)
            if len(stories) == POOL_SIZE:
                break
    order = list(range(POOL_SIZE))
    random.Random(SPLIT_SEED).shuffle(order)
    n_train_pool = int(POOL_SIZE * TRAIN_FRAC)
    train_pool, val_pool = order[:n_train_pool], order[n_train_pool:]
    n_train = round(n_stories * TRAIN_FRAC)
    n_val = n_stories - n_train
    return [stories[i] for i in train_pool[:n_train]], [stories[i] for i in val_pool[:n_val]]


def tokenizer_path(vocab_size):
    return DATA_DIR / f"tok{vocab_size}.json"


def get_tokenizer(vocab_size):
    path = tokenizer_path(vocab_size)
    if not path.exists():
        train, _ = load_split(POOL_SIZE)
        train_tokenizer(train, vocab_size, path)
    return load_tokenizer(path)


def _pack(tok, stories):
    ids = []
    for seq in tok.encode_batch(stories):
        ids.extend(seq)
        ids.append(tok.eos_id)
    return np.array(ids, dtype=np.uint16)


def get_dataset(n_stories, vocab_size):
    """Returns (train_tokens, val_tokens, meta) as flat uint16 arrays of packed stories."""
    tok = get_tokenizer(vocab_size)
    cache = DATA_DIR / f"ds_{n_stories}_v{vocab_size}"
    if not (cache / "meta.json").exists():
        cache.mkdir(parents=True, exist_ok=True)
        train, val = load_split(n_stories)
        train_ids, val_ids = _pack(tok, train), _pack(tok, val)
        np.save(cache / "train.npy", train_ids)
        np.save(cache / "val.npy", val_ids)
        meta = {
            "n_stories": n_stories,
            "n_train_stories": len(train),
            "n_val_stories": len(val),
            "vocab_size": vocab_size,
            "train_tokens": int(len(train_ids)),
            "val_tokens": int(len(val_ids)),
            # bytes per token on val, used to convert nats/token into bits-per-byte
            "val_bytes": sum(len(s.encode("utf-8")) + 1 for s in val),
        }
        (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    meta = json.loads((cache / "meta.json").read_text())
    return np.load(cache / "train.npy"), np.load(cache / "val.npy"), meta


def bits_per_byte(loss_nats_per_token, meta):
    return loss_nats_per_token / math.log(2) * meta["val_tokens"] / meta["val_bytes"]


def make_windows(tokens, seq_len):
    """Non-overlapping (seq_len + 1)-token windows over the packed stream."""
    n = (len(tokens) - 1) // seq_len
    return np.stack([tokens[i * seq_len : i * seq_len + seq_len + 1] for i in range(n)]).astype(np.int64)
