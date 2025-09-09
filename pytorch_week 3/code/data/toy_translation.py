"""
Toy parallel dataset generator.
Generates simple mapping pairs for quick experimentation. For example:
- English-ish sequences mapped to token-shifted 'foreign' sequences
- Or reversed words mapping (a learnable alignment)
This tiny generator returns tokenized pairs and builds small vocabularies.
"""

import random
from typing import List, Tuple, Dict
import os
import json

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

def build_vocab(sentences, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for s in sentences:
        for tok in s.split():
            cnt[tok] += 1
    vocab = [PAD, SOS, EOS, UNK] + [w for w, c in cnt.items() if c >= min_freq]
    idx = {w: i for i, w in enumerate(vocab)}
    return vocab, idx


def generate_synthetic_pair(n_samples=5000, max_len=6):
    """
    Create english->target pairs. Target can be:
      - english words reversed
      - english words with 'langX_' prefix to each token (simulates translation)
    """
    base_words = ["i", "you", "we", "they", "he", "she", "love", "hate", "see",
                  "like", "want", "play", "eat", "go", "school", "home", "book", "read"]
    pairs = []
    for _ in range(n_samples):
        L = random.randint(2, max_len)
        sent = " ".join(random.choices(base_words, k=L))
        # make target: reversed order + prefix
        tgt = " ".join([f"t_{w}" for w in sent.split()[::-1]])
        pairs.append((sent, tgt))
    return pairs


def tokenize(sent, vocab_idx):
    return [vocab_idx.get(tok, vocab_idx[UNK]) for tok in sent.split()]


def build_dataset(pairs, src_idx, tgt_idx, max_len=10):
    data = []
    for s, t in pairs:
        s_idx = [src_idx[SOS]] + [src_idx.get(tok, src_idx[UNK]) for tok in s.split()] + [src_idx[EOS]]
        t_idx = [tgt_idx[SOS]] + [tgt_idx.get(tok, tgt_idx[UNK]) for tok in t.split()] + [tgt_idx[EOS]]
        # pad to max_len
        s_idx = s_idx[:max_len] + [src_idx[PAD]] * max(0, max_len - len(s_idx))
        t_idx = t_idx[:max_len] + [tgt_idx[PAD]] * max(0, max_len - len(t_idx))
        data.append((s_idx, t_idx))
    return data


def make_and_save_dataset(outdir='data/mt', n_samples=2000, val_frac=0.1):
    os.makedirs(outdir, exist_ok=True)
    pairs = generate_synthetic_pair(n_samples)
    src_sents = [s for s, _ in pairs]
    tgt_sents = [t for _, t in pairs]
    src_vocab, src_idx = build_vocab(src_sents)
    tgt_vocab, tgt_idx = build_vocab(tgt_sents)
    max_len = 12
    data = build_dataset(pairs, {w:i for i,w in enumerate(src_vocab)}, {w:i for i,w in enumerate(tgt_vocab)}, max_len=max_len)

    # split
    split = int(len(data)*(1-val_frac))
    train = data[:split]
    val = data[split:]
    # save vocabs and splits
    with open(os.path.join(outdir, 'src_vocab.json'), 'w') as f:
        json.dump(src_vocab, f)
    with open(os.path.join(outdir, 'tgt_vocab.json'), 'w') as f:
        json.dump(tgt_vocab, f)
    import pickle
    with open(os.path.join(outdir, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(outdir, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    print(f"Saved dataset in {outdir} (train={len(train)}, val={len(val)})")
    return outdir
