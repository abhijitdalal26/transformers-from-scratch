"""Why does BPE overfit more than WordLevel on this corpus?

Three measurements, all cheap, all on the same seeded split:

1. **Tokens per word** - how many more prediction events the model has to make
   per sentence under each tokenizer. The training loss is an average over
   prediction events, so this changes what a "loss of 0.74" even means.

2. **Unigram entropy of the target** - the loss a model that learned *nothing*
   but the frequency of each piece would get (nats per token). This is the
   floor. A model whose held-out loss is near this floor has learned only
   unigram statistics, i.e. nothing transferable.

3. **Total target tokens seen per epoch** - the number of free supervision
   events per epoch, i.e. how much there is to memorise.

The point of the numbers: comparing the two variants on *loss* alone is
meaningless (different token granularity). The comparison that means something
is: **held-out loss as a fraction of the unigram floor**, and **how much
memorisable surface each variant gets per epoch**.

The learning code is not modified; this file only calls it.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import random_split

from _paths import PROJECT_DIR  # noqa: F401  (must precede the project imports)
import eval_variants as ev
from config import get_config
from tokenizers import Tokenizer

BPE_DIR = PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe"
OUT_DIR = PROJECT_DIR / "results" / "comparison" / "bpe_sweep"


def unigram_entropy(counts, n):
    """Shannon entropy in nats of the unigram distribution over `counts`."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", type=int, default=8000,
                   help="how many training sentences to measure (default 8000)")
    p.add_argument("--out", default=str(OUT_DIR / "tokenizer_stats.json"))
    args = p.parse_args()

    config = get_config()
    from datasets import load_dataset
    ds_raw = load_dataset(config["datasource"],
                           f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    seed = config.get("seed", 42)
    train_size = int(0.9 * len(ds_raw))
    train_raw, _ = random_split(
        ds_raw, [train_size, len(ds_raw) - train_size],
        generator=torch.Generator().manual_seed(seed))

    # NB: slicing a HF Dataset returns a dict of columns, not a list of rows,
    # so index it one row at a time.
    n = min(args.sentences, len(train_raw))
    srcs = [train_raw[i]["translation"][config["lang_src"]] for i in range(n)]
    tgts = [train_raw[i]["translation"][config["lang_tgt"]] for i in range(n)]

    out = {}
    for variant in ("wordlevel", "bpe"):
        tok_dir = ev.TOKENIZER_DIRS[variant]
        ts = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
        tt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))

        src_tok = [ts.encode(s).ids for s in srcs]
        tgt_tok = [tt.encode(t).ids for t in tgts]

        src_words = sum(len(s.split()) for s in srcs)
        tgt_words = sum(len(t.split()) for t in tgts)
        src_toks = sum(len(t) for t in src_tok)
        tgt_toks = sum(len(t) for t in tgt_tok)

        tgt_counts = Counter(i for t in tgt_tok for i in t)
        src_counts = Counter(i for t in src_tok for i in t)

        out[variant] = {
            "sentences_measured": len(tgts),
            "vocab_src": ts.get_vocab_size(),
            "vocab_tgt": tt.get_vocab_size(),
            "src_tokens": src_toks,
            "src_words": src_words,
            "tgt_tokens": tgt_toks,
            "tgt_words": tgt_words,
            "src_tokens_per_word": round(src_toks / max(src_words, 1), 4),
            "tgt_tokens_per_word": round(tgt_toks / max(tgt_words, 1), 4),
            "src_unigram_entropy_nats_per_token": round(unigram_entropy(src_counts, src_toks), 4),
            "tgt_unigram_entropy_nats_per_token": round(unigram_entropy(tgt_counts, tgt_toks), 4),
            # the same entropy expressed per word, which is comparable across
            # variants (one word = one prediction event in the word-level model)
            "tgt_unigram_entropy_nats_per_word": round(
                unigram_entropy(tgt_counts, tgt_toks) * tgt_toks / max(tgt_words, 1), 4),
            "tgt_distinct_tokens_seen": len(tgt_counts),
            "tgt_distinct_words_seen": len(set(w for t in tgts for w in t.split())),
        }
        print("%-10s vocab %5d/%-5d  src tok/word %.3f  tgt tok/word %.3f  "
              "tgt unigram entropy %.3f nats/token (%.3f nats/word)"
              % (variant, ts.get_vocab_size(), tt.get_vocab_size(),
                 out[variant]["src_tokens_per_word"], out[variant]["tgt_tokens_per_word"],
                 out[variant]["tgt_unigram_entropy_nats_per_token"],
                 out[variant]["tgt_unigram_entropy_nats_per_word"]), flush=True)

    out["notes"] = {
        "tokens_per_epoch_wordlevel": out["wordlevel"]["tgt_tokens"],
        "tokens_per_epoch_bpe": out["bpe"]["tgt_tokens"],
        "extra_supervision_events_bpe": out["bpe"]["tgt_tokens"] - out["wordlevel"]["tgt_tokens"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
