"""Overfitting diagnostic: teacher-forced loss on TRAIN vs VAL, per token and per word.

Why this exists
---------------
BLEU alone cannot tell you *why* a model is worse. This measures the model's
loss on data it memorised (train) against data it has never seen (val):

  * train loss low  + val loss high  -> overfitting (the model memorised)
  * both high                          -> the model never learned / the saved
                                         tokenizer does not match training

Per-word numbers are the ones to compare across tokenizer variants, because the
WordLevel model emits 1 token per word while the BPE model emits ~1.4 pieces
per word. Comparing per-token numbers across variants is meaningless.

Usage
-----
    python tools/overfit_diagnostic.py --variant bpe \
        --model-folder drive_bpe/transformers-from-scratch_weights_bpe \
        --checkpoint drive_bpe/transformers-from-scratch_weights_bpe/tmodel_17.pt
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["wordlevel", "bpe"], default="wordlevel")
    p.add_argument("--model-folder", default=None, help="folder holding the checkpoint + tokenizers")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples", type=int, default=200, help="sentences from each split")
    p.add_argument("--json", default=None, help="write the result to this json file")
    return p.parse_args()


def resolve_paths(args):
    """Where the checkpoints and the tokenizers of each variant live."""
    if args.model_folder:
        folder = str(Path(args.model_folder).resolve())
        return folder, os.path.join(folder, "tokenizer_{0}.json")
    if args.variant == "bpe":
        folder = str((PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe").resolve())
    else:
        folder = str((PROJECT_DIR / "model").resolve())
    return folder, os.path.join(folder, "tokenizer_{0}.json")


def measure(model, loader, tok_tgt, n_sentences):
    """Teacher-forced CE (no label smoothing) over the non-pad target positions."""
    pad = tok_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
    nats = 0.0
    n_tok = 0
    n_words = 0
    seen = 0
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
    args = parse_args()
    os.chdir(PROJECT_DIR)
    model_folder, tokenizer_file = resolve_paths(args)

    # config.get_config() honours these two env vars (added for the BPE run,
    # whose artifacts live in a different folder than the default model/).
    os.environ["MODEL_FOLDER"] = model_folder
    os.environ["TOKENIZER_FILE"] = tokenizer_file

    import config as cfgmod
    import train as tr

    cfg = cfgmod.get_config()
    train_loader, val_loader, tok_src, tok_tgt = tr.get_ds(cfg)

    model = tr.get_model(cfg, tok_src.get_vocab_size(), tok_tgt.get_vocab_size())
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # the val loader has batch_size 1; the train loader has batch_size 8
    val = measure(model, val_loader, tok_tgt, args.samples)
    train = measure(model, train_loader, tok_tgt, args.samples)

    print(f"checkpoint        : {args.checkpoint}")
    print(f"variant           : {args.variant}")
    print(f"target vocab      : {tok_tgt.get_vocab_size()}")
    print(f"sentences per split: {args.samples}")
    print()
    print(f"{'split':<6} {'nats/token':>11} {'nats/word':>10} {'tokens':>8} {'words':>7}")
    print(f"{'train':<6} {train[0]:>11.3f} {train[1]:>10.3f} {train[2]:>8d} {train[3]:>7d}")
    print(f"{'val':<6} {val[0]:>11.3f} {val[1]:>10.3f} {val[2]:>8d} {val[3]:>7d}")
    print()
    print(f"generalisation gap : {val[0] - train[0]:.3f} nats/token "
          f"({val[1] - train[1]:.3f} nats/word)")

    if args.json:
        import json

        payload = {
            "variant": args.variant,
            "checkpoint": str(args.checkpoint),
            "samples_per_split": args.samples,
            "train_nats_per_token": round(train[0], 4),
            "val_nats_per_token": round(val[0], 4),
            "gap_nats_per_token": round(val[0] - train[0], 4),
            "train_nats_per_word": round(train[1], 4),
            "val_nats_per_word": round(val[1], 4),
            "gap_nats_per_word": round(val[1] - train[1], 4),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
