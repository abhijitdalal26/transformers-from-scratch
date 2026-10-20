"""Show what the models actually output, side by side.

`translate.py` translates one sentence with one checkpoint. This translates a
set of sentences with several checkpoints in one run and prints them in a
readable table, so "the model memorised" can be checked by reading text
instead of only by reading a score.

The sentences come in two kinds:

  * **held-out** - sentences from the seeded validation split (never trained
    on, but from the same books as the training data),
  * **never seen** - short sentences written from scratch, in neither split
    (`tools/novel_sentences.py`).

Usage:
    python tools/show_inference.py --specs bpe:tmodel_10.pt bpe:tmodel_17.pt \
        wordlevel:tmodel_17.pt --val 5
"""

import argparse
import json
from pathlib import Path

import torch

from _paths import PROJECT_DIR  # noqa: F401  (must precede the project imports)
import eval_variants as ev
from config import get_config
from model import build_transformer
from tokenizers import Tokenizer

BPE_DIR = PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe"
WL_DIR = PROJECT_DIR / "model"
OUT_DIR = PROJECT_DIR / "results" / "comparison" / "bpe_sweep"


def ckpt_path(variant, name):
    return (BPE_DIR if variant == "bpe" else WL_DIR) / name


def load(variant, path, config, device):
    tok_dir = ev.TOKENIZER_DIRS[variant]
    ts = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
    tt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))
    model = build_transformer(ts.get_vocab_size(), tt.get_vocab_size(),
                              config["seq_len"], config["seq_len"],
                              d_model=config["d_model"]).to(device)
    state = torch.load(str(path), map_location=device, weights_only=False, mmap=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, ts, tt


def translate_batch(model, ts, tt, sentences, config, device):
    """Greedy decode a list of English sentences (same decoder as training)."""
    from train import greedy_decode
    seq_len = config["seq_len"]
    pad = ts.token_to_id("[PAD]")
    out = []
    for en in sentences:
        ids = ts.encode(en).ids[:seq_len - 2]
        source = torch.cat([
            torch.tensor([ts.token_to_id("[SOS]")], dtype=torch.int64),
            torch.tensor(ids, dtype=torch.int64),
            torch.tensor([ts.token_to_id("[EOS]")], dtype=torch.int64),
            torch.tensor([pad] * (seq_len - len(ids) - 2), dtype=torch.int64),
        ]).unsqueeze(0).to(device)
        mask = (source != pad).unsqueeze(1).unsqueeze(1).int().to(device)
        gen = greedy_decode(model, source, mask, None, tt, seq_len, device)
        out.append(tt.decode(gen.detach().cpu().numpy()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", required=True)
    p.add_argument("--val", type=int, default=5, help="how many held-out sentences to show")
    p.add_argument("--novel", type=int, default=8, help="how many never-seen sentences")
    p.add_argument("--out", default=str(OUT_DIR / "inference_showcase.md"))
    args = p.parse_args()

    config = get_config()
    device = torch.device("cpu")

    # the held-out sentences (identical for every checkpoint)
    tok_dir = ev.TOKENIZER_DIRS["bpe"]
    ts = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
    tt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))
    val_ds = ev.build_val_dataset(config, ts, tt)
    val_items = [val_ds[i] for i in range(min(args.val, len(val_ds)))]
    val_src = [it["src_text"] for it in val_items]
    val_ref = [it["tgt_text"] for it in val_items]

    novel_src = [s[0] for s in ev.SENTENCES[:args.novel]]
    novel_ref = [s[1] for s in ev.SENTENCES[:args.novel]]

    labels, outputs = [], {}
    for spec in args.specs:
        variant, _, name = spec.partition(":")
        path = ckpt_path(variant, name)
        if not path.exists():
            print("MISSING %s" % path)
            continue
        label = "%s ep%s" % (variant, path.stem.split("_")[-1].lstrip("0"))
        model, ts_v, tt_v = load(variant, path, config, device)
        outputs[label] = {
            "val": translate_batch(model, ts_v, tt_v, val_src, config, device),
            "novel": translate_batch(model, ts_v, tt_v, novel_src, config, device),
        }
        labels.append(label)
        print("decoded with", label, flush=True)
        del model

    lines = ["# What the models actually say", "",
             "Greedy decoding, same sentences for every checkpoint.", ""]

    lines += ["## Never-seen sentences (in neither train nor val)", "",
              "| English | Reference | " + " | ".join("`%s`" % l for l in labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    for i, (en, ref) in enumerate(zip(novel_src, novel_ref)):
        cells = [outputs[l]["novel"][i].replace("|", "\\|") for l in labels]
        lines.append("| %s | %s | %s |" % (en, ref, " | ".join(cells)))
    lines.append("")

    lines += ["## Held-out sentences from the validation split", "",
              "| English | Reference | " + " | ".join("`%s`" % l for l in labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    for i, (en, ref) in enumerate(zip(val_src, val_ref)):
        cells = [outputs[l]["val"][i].replace("|", "\\|") for l in labels]
        lines.append("| %s | %s | %s |" % (en[:150], ref[:150], " | ".join(cells)))
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("wrote", out)

    # also print it, so it can be read straight away
    print("\n".join(lines))


if __name__ == "__main__":
    main()
