"""Does the model actually use the source sentence?

Teacher-forced loss on the SAME held-out sentences, with three different
encoder inputs:

  real      - the actual English sentence
  shuffled  - the same number of tokens, but token ids sampled at random from
              the source vocabulary (plausible-looking Dutch/English-looking
              noise, correct length, so the model still sees a well-formed
              sequence it cannot read)
  blank     - every source position is [PAD], i.e. the encoder sees nothing at
              all and can only behave like a language model

If `shuffled` and `blank` cost the model almost as much as `real`, the decoder
is not translating - it is completing text. That is the difference between a
model that learned a conditional distribution P(target | source) and one that
learned a marginal language model P(target) with a translation-shaped decoder
around it.

This is the cheapest way to separate "the model memorised the training set"
from "the model never learned to use the source", and the two need different
fixes.

The learning code is not modified; this file only calls it.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from _paths import PROJECT_DIR  # noqa: F401  (must precede the project imports)
import eval_variants as ev
from config import get_config
from dataset import BilingualDataset
from model import build_transformer
from tokenizers import Tokenizer
from torch.utils.data import random_split

BPE_DIR = PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe"
WL_DIR = PROJECT_DIR / "model"
OUT_DIR = PROJECT_DIR / "results" / "comparison" / "bpe_sweep"


def ckpt_path(variant, name):
    return (BPE_DIR if variant == "bpe" else WL_DIR) / name


def build_val(config, ts, tt, n):
    from datasets import load_dataset
    ds_raw = load_dataset(config["datasource"],
                           f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    seed = config.get("seed", 42)
    train_size = int(0.9 * len(ds_raw))
    _, val_raw = random_split(
        ds_raw, [train_size, len(ds_raw) - train_size],
        generator=torch.Generator().manual_seed(seed))
    val_ds = BilingualDataset(val_raw, ts, tt, config["lang_src"],
                              config["lang_tgt"], config["seq_len"])
    return DataLoader(Subset(val_ds, range(min(n, len(val_ds)))),
                      batch_size=8, shuffle=False)


def loss_on(model, loader, tok_tgt, mode, vocab_src, n_sentences):
    pad = tok_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    nats = 0.0
    n_tok = 0
    n_words = 0
    seen = 0
    g = torch.Generator().manual_seed(1234)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if seen >= n_sentences:
                break
            enc = batch["encoder_input"]
            mask = batch["encoder_mask"]
            if mode == "shuffled":
                # same shape, random token ids from the source vocabulary
                enc = torch.randint(0, vocab_src, enc.shape, generator=g)
                mask = (enc != pad).unsqueeze(1).unsqueeze(1).int()
            elif mode == "blank":
                enc = torch.full_like(enc, pad)
                mask = torch.zeros_like(mask)

            encoder_output = model.encode(enc, mask)
            decoder_output = model.decode(encoder_output, mask,
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
    return nats / max(n_tok, 1), nats / max(n_words, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--out", default=str(OUT_DIR / "source_ablation.json"))
    args = p.parse_args()

    config = get_config()
    device = torch.device("cpu")
    rows = []
    cache = {}
    for spec in args.specs:
        variant, _, name = spec.partition(":")
        path = ckpt_path(variant, name)
        if not path.exists():
            print("MISSING %s" % path, flush=True)
            continue
        if variant not in cache:
            tok_dir = ev.TOKENIZER_DIRS[variant]
            ts = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
            tt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))
            cache[variant] = (ts, tt, build_val(config, ts, tt, args.samples))
        ts, tt, loader = cache[variant]

        model = build_transformer(ts.get_vocab_size(), tt.get_vocab_size(),
                                  config["seq_len"], config["seq_len"],
                                  d_model=config["d_model"]).to(device)
        state = torch.load(str(path), map_location=device, weights_only=False, mmap=True)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        del state

        real = loss_on(model, loader, tt, "real", ts.get_vocab_size(), args.samples)
        shuf = loss_on(model, loader, tt, "shuffled", ts.get_vocab_size(), args.samples)
        blank = loss_on(model, loader, tt, "blank", ts.get_vocab_size(), args.samples)

        row = {
            "variant": variant,
            "checkpoint": path.name,
            "epoch": path.stem.split("_")[-1],
            "real_nats_per_token": round(real[0], 4),
            "shuffled_nats_per_token": round(shuf[0], 4),
            "blank_nats_per_token": round(blank[0], 4),
            "real_nats_per_word": round(real[1], 4),
            "shuffled_nats_per_word": round(shuf[1], 4),
            "blank_nats_per_word": round(blank[1], 4),
            # how much of the loss is explained by the source sentence:
            # (blank - real) / blank
            "source_dependence": round((blank[0] - real[0]) / max(blank[0], 1e-9), 4),
        }
        rows.append(row)
        print("%-10s ep%-3s  real %.3f | shuffled %.3f | blank %.3f nats/token   "
              "source dependence %.1f%%"
              % (variant, row["epoch"], row["real_nats_per_token"],
                 row["shuffled_nats_per_token"], row["blank_nats_per_token"],
                 row["source_dependence"] * 100), flush=True)
        del model
        gc.collect()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out, flush=True)

    md = [
        "# Source ablation - is the model using the English sentence?",
        "",
        "Teacher-forced loss on the same %d held-out sentences, changing only the "
        "encoder input." % args.samples,
        "",
        "| variant | epoch | real source | shuffled source | blank source | "
        "**source dependence** |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append("| %s | %s | %.3f | %.3f | %.3f | **%.1f%%** |"
                  % (r["variant"], r["epoch"], r["real_nats_per_token"],
                     r["shuffled_nats_per_token"], r["blank_nats_per_token"],
                     r["source_dependence"] * 100))
    md += ["",
           "`source dependence` = (blank - real) / blank: how much of the model's "
           "held-out prediction the English sentence is actually worth.",
           ""]
    md_out = out.with_suffix(".md")
    md_out.write_text("\n".join(md))
    print("wrote", md_out, flush=True)


if __name__ == "__main__":
    main()
