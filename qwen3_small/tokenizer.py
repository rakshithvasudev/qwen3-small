"""Byte-level BPE tokenizer trained on the TinyStories training split.

Byte-level BPE never produces an unknown token: any word missing from the
vocabulary falls back to smaller pieces, down to single bytes. This matters
for the classifier text, which uses words that never appear in TinyStories.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<eos>"
PAD = "<pad>"


def train_tokenizer(texts, vocab_size, path):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))
    return load_tokenizer(path)


class StoryTokenizer:
    def __init__(self, tok):
        self.tok = tok
        self.eos_id = tok.token_to_id(EOS)
        self.pad_id = tok.token_to_id(PAD)
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def load_tokenizer(path):
    return StoryTokenizer(Tokenizer.from_file(str(path)))
