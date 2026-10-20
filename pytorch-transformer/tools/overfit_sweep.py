"""Teacher-forced train-vs-val loss for a whole list of checkpoints.

`overfit_diagnostic.py` measures one checkpoint. This measures many, in one
process, so the *shape* of the generalisation gap over the epochs is visible:
the whole point of the overfitting question is not "is the gap big at epoch 17"
but "how does the gap grow while the validation score stops moving".

Two differences from `overfit_diagnostic.py`, both deliberate:

1. **The train subset is fixed.** `train.get_ds()` builds the training
   DataLoader with `shuffle=True`, so `overfit_diagnostic.py` measures a
   *different* 200 training sentences on every run - the numbers were not
   comparable between checkpoints. Here the first N sentences of the seeded
   train split are used, in order, for every checkpoint.

2. **The split is the training split, not a random subset of it.** The
   tokenizer is built from the whole corpus in `train.py` (a known leak that
   favours BPE), so "train" here means "sentences the model was trained on".

The learning code is not modified; this file only calls it.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from _paths import PROJECT_DIR  # noqa: F401  (must precede the project imports)
import eval_variants as ev
from config import get_config
from dataset import BilingualDataset
from model import build_transformer
from tokenizers import Tokenizer
from torch.utils.data import random_split, DataLoader, Subset

BPE_DIR = PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe"
WL_DIR = PROJECT_DIR / "model"
OUT_DIR = PROJECT_DIR / "results" / "comparison" / "bpe_sweep"


def ckpt_path(variant, name):
    return (BPE_DIR if variant == "bpe" else WL_DIR) / name


def build_splits(config, ts, tt):
    """The seeded train/val split, plus a deterministic slice of each."""
    from datasets import load_dataset
    ds_raw = load_dataset(config["datasource"],
                           f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    seed = config.get("seed", 42)
    train_size = int(0.9 * len(ds_raw))
    train_raw, val_raw = random_split(
        ds_raw, [train_size, len(ds_raw) - train_size],
        generator=torch.Generator().manual_seed(seed))
    train_ds = BilingualDataset(train_raw, ts, tt, config["lang_src"],
                                config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_raw, ts, tt, config["lang_src"],
                              config["lang_tgt"], config["seq_len"])
    return train_ds, val_ds


def measure(model, loader, tok_tgt, n_sentences, batch_size):
    """Teacher-forced CE (no label smoothing) over the non-pad target positions."""
    pad = tok_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    nats = 0.0
    n_tok = 0
    n_words = 0
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if seen >= n_sentences:
                break
            encoder_output = model.encode(batch["encoder_input"], batch["encoder_mask"])
            decoder_output = model.decode(encoder_output, batch["encoder_mask"],
                                          batch["decoder_input"], batch["decoder_mask"])
            proj = model.project(decoder_output)
            label = batch["label"]
            loss = loss_fn(proj.view(-1, proj.size(-1)), label.view(-1))
            n = (label != pad).sum().item()
            nats += loss.item() * n
            n_tok += n
            txt = batch["tgt_text"]
            for t in (txt if isinstance(txt, list) else [txt]):
                n_words += len(t.split())
            seen += label.size(0)
    return nats / max(n_tok, 1), nats / max(n_words, 1), n_tok, n_words


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", required=True,
                   help="variant:name entries, e.g. bpe:tmodel_10.pt")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=str(OUT_DIR / "overfit_sweep.json"))
    args = p.parse_args()

    config = get_config()
    device = torch.device("cpu")
    rows = []

    # the tokenizers are the same for every checkpoint of a variant, so the
    # dataset is built once per variant
    cache = {}
    for spec in args.specs:
        variant, _, name = spec.partition(":")
        path = ckpt_path(variant, name)
        if not path.exists():
            print("MISSING %s - skipped" % path, flush=True)
            continue
        if variant not in cache:
            tok_dir = ev.TOKENIZER_DIRS[variant]
            ts = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
            tt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))
            train_ds, val_ds = build_splits(config, ts, tt)
            # deterministic, ordered slice of the train split
            train_loader = DataLoader(Subset(train_ds, range(min(args.samples, len(train_ds)))),
                                      batch_size=args.batch_size, shuffle=False)
            val_loader = DataLoader(Subset(val_ds, range(min(args.samples, len(val_ds)))),
                                    batch_size=args.batch_size, shuffle=False)
            cache[variant] = (ts, tt, train_loader, val_loader)
        ts, tt, train_loader, val_loader = cache[variant]

        model = build_transformer(ts.get_vocab_size(), tt.get_vocab_size(),
                                  config["seq_len"], config["seq_len"],
                                  d_model=config["d_model"]).to(device)
        state = torch.load(str(path), map_location=device, weights_only=False, mmap=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        del state

        val = measure(model, val_loader, tt, args.samples, args.batch_size)
        tr = measure(model, train_loader, tt, args.samples, args.batch_size)

        row = {
            "variant": variant,
            "checkpoint": path.name,
            "epoch": path.stem.split("_")[-1],
            "train_nats_per_token": round(tr[0], 4),
            "val_nats_per_token": round(val[0], 4),
            "gap_nats_per_token": round(val[0] - tr[0], 4),
            "train_nats_per_word": round(tr[1], 4),
            "val_nats_per_word": round(val[1], 4),
            "gap_nats_per_word": round(val[1] - tr[1], 4),
            "tokens": val[2],
            "words": val[3],
        }
        rows.append(row)
        print("%-8s ep%-3s  train %.3f  val %.3f  gap %.3f nats/token | "
              "train %.3f  val %.3f  gap %.3f nats/word"
              % (variant, row["epoch"], row["train_nats_per_token"],
                 row["val_nats_per_token"], row["gap_nats_per_token"],
                 row["train_nats_per_word"], row["val_nats_per_word"],
                 row["gap_nats_per_word"]), flush=True)

        del model
        gc.collect()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out, flush=True)

    # markdown table next to the json
    md = [
        "# Generalisation gap per checkpoint (teacher-forced loss)",
        "",
        "- train subset = the first %d sentences of the seeded train split (fixed, "
        "not reshuffled), val = the first %d sentences of the seeded val split"
        % (args.samples, args.samples),
        "- loss = cross-entropy in nats per token / per word, no label smoothing, "
        "padding excluded",
        "",
        "| variant | epoch | train nats/token | val nats/token | **gap** | "
        "train nats/word | val nats/word | **gap** |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append("| %s | %s | %.3f | %.3f | **%.3f** | %.3f | %.3f | **%.3f** |"
                  % (r["variant"], r["epoch"], r["train_nats_per_token"],
                     r["val_nats_per_token"], r["gap_nats_per_token"],
                     r["train_nats_per_word"], r["val_nats_per_word"],
                     r["gap_nats_per_word"]))
    md.append("")
    md_out = out.with_suffix(".md")
    md_out.write_text("\n".join(md))
    print("wrote", md_out, flush=True)


if __name__ == "__main__":
    main()
