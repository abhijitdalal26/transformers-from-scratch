"""Evaluate every available BPE checkpoint with ONE identical protocol.

`eval_variants.py` already defines the protocol (same seeded held-out split,
same greedy decoder, same corpus BLEU/CER/WER). This tool reuses those exact
functions - `build_val_dataset`, `decode_val`, `score_block` - so the numbers
it prints are directly comparable with the ones already stored in
`results/comparison/eval_*.json`, but it adds:

  * a checkpoint sweep (every epoch on disk, not a hand-picked few),
  * memory-mapped checkpoint loading, so several shards can run in parallel
    on one machine,
  * a per-shard JSON dump plus a merged markdown table,
  * the decoded sentences, so the numbers can be read against the text.

The learning code (`model.py`, `dataset.py`, `train.py`, `config.py`,
`translate.py`) is untouched - this file only calls it.

Usage:
    # one shard
    python tools/bpe_sweep.py --specs bpe:tmodel_09.pt bpe:tmodel_10.pt

    # three shards in parallel, then merge
    python tools/bpe_sweep.py --shard 0 --shards 3 --all
    python tools/bpe_sweep.py --merge
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from _paths import PROJECT_DIR  # noqa: F401  (must precede the project imports)
import eval_variants as ev
from config import get_config
from model import build_transformer
from tokenizers import Tokenizer

BPE_DIR = PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe"
WL_DIR = PROJECT_DIR / "model"

# every checkpoint that is actually on disk, in epoch order
BPE_CKPTS = ["tmodel_09.pt", "tmodel_10.pt", "tmodel_12.pt",
             "tmodel_16.pt", "tmodel_17.pt", "tmodel_18.pt", "tmodel_19.pt"]
WL_CKPTS = ["tmodel_17.pt"]

OUT_DIR = PROJECT_DIR / "results" / "comparison" / "bpe_sweep"


def ckpt_path(variant, name):
    base = BPE_DIR if variant == "bpe" else WL_DIR
    return base / name


def load_model(variant, path, config, device):
    """Same model as eval_variants.load_model, but memory-mapped.

    The checkpoints carry the Adam moments as well (~679 MB each). mmap=True
    keeps them on disk instead of in RAM, which is what makes it safe to run
    several shards of this sweep in parallel.
    """
    tok_dir = ev.TOKENIZER_DIRS[variant]
    ts = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
    tt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))
    model = build_transformer(
        ts.get_vocab_size(), tt.get_vocab_size(),
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)
    state = torch.load(str(path), map_location=device, weights_only=False, mmap=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    info = {
        "epoch": state["epoch"],
        "global_step": state.get("global_step"),
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "vocab_src": ts.get_vocab_size(),
        "vocab_tgt": tt.get_vocab_size(),
    }
    del state
    return model, ts, tt, info


def evaluate_one(variant, path, config, device, n_samples, want_novel=True):
    t0 = time.time()
    model, ts, tt, info = load_model(variant, path, config, device)
    val_ds = ev.build_val_dataset(config, ts, tt)
    src, ref, pred, gen_len = ev.decode_val(model, tt, val_ds, n_samples, config, device)

    cased = ev.score_block(pred, ref)
    blind = ev.score_block([p.lower() for p in pred], [r.lower() for r in ref])

    novel_out = []
    if want_novel:
        novel_pred = ev.decode_novel(model, ts, tt, config, device)
        novel_out = [{"en": en, "ref": nl, "out": o}
                     for (en, nl, _), o in zip(ev.SENTENCES, novel_pred)]

    res = {
        "variant": variant,
        "checkpoint": str(path),
        "checkpoint_name": path.name,
        "checkpoint_epoch": info["epoch"],
        "global_step": info["global_step"],
        "params_M": info["params_M"],
        "vocab_src": info["vocab_src"],
        "vocab_tgt": info["vocab_tgt"],
        "n_sentences": len(pred),
        "cased": cased,
        "case_blind": blind,
        "mean_gen_len_tokens": round(sum(gen_len) / max(len(gen_len), 1), 2),
        "mean_gen_len_words": round(sum(len(p.split()) for p in pred) / max(len(pred), 1), 2),
        "mean_ref_len_words": round(sum(len(r.split()) for r in ref) / max(len(ref), 1), 2),
        "src_unk_rate": round(ev.unk_rate(src, ts), 6),
        "ref_unk_rate_tgt": round(ev.unk_rate(ref, tt), 6),
        "repeated_4gram_sentences": ev.repeated_4gram_sentences(pred),
        "empty_predictions": sum(1 for p in pred if not p.strip()),
        "decode_seconds_total": round(time.time() - t0, 1),
        "novel_outputs": novel_out,
        "novel_hits_ge50pct": ev.novel_hits(
            [o["out"] for o in novel_out]) if novel_out else None,
        "_src": src, "_ref": ref, "_pred": pred,
    }
    del model
    gc.collect()
    return res


def specs_from_args(args):
    if args.all:
        out = [("bpe", n) for n in BPE_CKPTS] + [("wordlevel", n) for n in WL_CKPTS]
    else:
        out = []
        for s in args.specs:
            variant, _, name = s.partition(":")
            if not name:
                variant, name = "bpe", variant
            out.append((variant, name))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="*", default=[])
    p.add_argument("--all", action="store_true", help="sweep every checkpoint on disk")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--no-novel", action="store_true")
    p.add_argument("--merge", action="store_true", help="merge shard results into one table")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge(out_dir)
        return

    config = get_config()
    device = torch.device("cpu")
    specs = specs_from_args(args)
    mine = specs[args.shard::args.shards]
    print("shard %d/%d -> %s" % (args.shard, args.shards,
                                 ", ".join("%s:%s" % s for s in mine)), flush=True)

    for variant, name in mine:
        path = ckpt_path(variant, name)
        if not path.exists():
            print("  MISSING %s - skipped" % path, flush=True)
            continue
        res = evaluate_one(variant, path, config, device, args.samples,
                            want_novel=not args.no_novel)
        tag = "%s_%s" % (variant, path.stem)
        (out_dir / ("eval_%s.json" % tag)).write_text(json.dumps(res, indent=2))
        print("  %-22s ep%-3s cased BLEU %6.2f | case-blind %6.2f | CER %.3f | WER %.3f | novel %s/20"
              % (tag, res["checkpoint_epoch"], res["cased"]["bleu4"],
                 res["case_blind"]["bleu4"], res["case_blind"]["cer"],
                 res["case_blind"]["wer"], res["novel_hits_ge50pct"]), flush=True)

    print("shard done", flush=True)


def merge(out_dir):
    files = sorted(out_dir.glob("eval_*.json"))
    if not files:
        raise SystemExit("no shard results in %s" % out_dir)
    results = []
    for f in files:
        r = json.loads(f.read_text())
        results.append(r)
    order = {"bpe": 0, "wordlevel": 1}
    results.sort(key=lambda r: (order.get(r["variant"], 9), r["checkpoint_epoch"]))

    labels = ["%s_ep%s" % (r["variant"], r["checkpoint_epoch"]) for r in results]
    lines = [
        "# BPE checkpoint sweep - every checkpoint on disk, one protocol",
        "",
        "- **Decoding:** greedy | **Split:** seeded (seed 42) 90/10 split of "
        "`Helsinki-NLP/opus_books` (en-nl) | **Sentences:** %d held-out"
        % results[0]["n_sentences"],
        "- **Case-blind** = prediction and reference lowercased before scoring "
        "(the BPE tokenizer lowercases its output, the WordLevel one does not)",
        "- Produced by `tools/bpe_sweep.py`",
        "",
        "## Scores",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]

    def row(name, getter):
        lines.append("| %s | " % name + " | ".join(str(getter(r)) for r in results) + " |")

    row("variant", lambda r: r["variant"])
    row("epoch", lambda r: r["checkpoint_epoch"])
    row("global_step", lambda r: r["global_step"])
    row("params (M)", lambda r: r["params_M"])
    row("vocab src / tgt", lambda r: "%d / %d" % (r["vocab_src"], r["vocab_tgt"]))
    row("**BLEU-4 (cased)**", lambda r: "%.2f" % r["cased"]["bleu4"])
    row("**BLEU-4 (case-blind)**", lambda r: "%.2f" % r["case_blind"]["bleu4"])
    row("BLEU-1 prec (cased)", lambda r: "%.4f" % r["cased"]["precisions"][0])
    row("BLEU-4 prec (cased)", lambda r: "%.4f" % r["cased"]["precisions"][3])
    row("brevity penalty", lambda r: "%.2f" % r["case_blind"]["bp"])
    row("CER (cased)", lambda r: "%.4f" % r["cased"]["cer"])
    row("WER (cased)", lambda r: "%.4f" % r["cased"]["wer"])
    row("CER (case-blind)", lambda r: "%.4f" % r["case_blind"]["cer"])
    row("WER (case-blind)", lambda r: "%.4f" % r["case_blind"]["wer"])
    row("exact match (cased)", lambda r: r["cased"]["exact"])
    row("mean gen len (tokens)", lambda r: r["mean_gen_len_tokens"])
    row("mean gen len (words)", lambda r: r["mean_gen_len_words"])
    row("mean ref len (words)", lambda r: r["mean_ref_len_words"])
    row("src [UNK] rate", lambda r: "%.4f" % r["src_unk_rate"])
    row("ref [UNK] rate", lambda r: "%.4f" % r["ref_unk_rate_tgt"])
    row("looped sentences (4-gram)", lambda r: r["repeated_4gram_sentences"])
    row("empty predictions", lambda r: r["empty_predictions"])
    row("never-seen hits >=50% ref", lambda r: r["novel_hits_ge50pct"])
    row("decode seconds", lambda r: r["decode_seconds_total"])

    best_cased = max(results, key=lambda r: r["cased"]["bleu4"])
    best_blind = max(results, key=lambda r: r["case_blind"]["bleu4"])
    bpe_rows = [r for r in results if r["variant"] == "bpe"]
    lines += [
        "",
        "## Verdict",
        "",
        "- **Best cased BLEU-4 (all variants):** `%s` (epoch %s) - %.2f"
        % (Path(best_cased["checkpoint"]).name, best_cased["checkpoint_epoch"],
           best_cased["cased"]["bleu4"]),
        "- **Best case-blind BLEU-4 (all variants):** `%s` (epoch %s) - %.2f"
        % (Path(best_blind["checkpoint"]).name, best_blind["checkpoint_epoch"],
           best_blind["case_blind"]["bleu4"]),
    ]
    if bpe_rows:
        bb = max(bpe_rows, key=lambda r: r["cased"]["bleu4"])
        lines += [
            "- **Best BPE checkpoint:** `%s` (epoch %s) - %.2f cased / %.2f case-blind"
            % (Path(bb["checkpoint"]).name, bb["checkpoint_epoch"],
               bb["cased"]["bleu4"], bb["case_blind"]["bleu4"]),
        ]
    lines += [""]

    # per-sentence head-to-head (case-blind)
    # only the checkpoints that were produced by this tool carry the decoded
    # sentences; the ones scored by eval_variants.py store only the metrics.
    import re
    from train import corpus_bleu
    scored = [r for r in results if r.get("_pred") and r.get("_ref")]
    labels_scored = ["%s_ep%s" % (r["variant"], r["checkpoint_epoch"]) for r in scored]
    per = []
    for r in scored:
        per.append([corpus_bleu([a.lower()], [b.lower()])[0]
                    for a, b in zip(r["_pred"], r["_ref"])])
    wins = {lab: 0 for lab in labels_scored}
    ties = 0
    for row_ in zip(*per):
        best_v = max(row_)
        if row_.count(best_v) > 1:
            ties += 1
        else:
            wins[labels_scored[row_.index(best_v)]] += 1
    lines += ["## Head-to-head per sentence (case-blind BLEU)", "",
              "Computed over the %d checkpoints that stored their decoded "
              "sentences." % len(scored), "",
              "| checkpoint | sentences won |", "|---|---|"]
    for lab in labels_scored:
        lines.append("| `%s` | %d |" % (lab, wins[lab]))
    lines += ["| tied | %d |" % ties, ""]

    # never-seen sentences
    lines += ["## Never-seen sentences (model output)", "",
              "| English | Reference | " + " | ".join("`%s`" % l for l in labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    for i, (en, nl, _) in enumerate(ev.SENTENCES):
        cells = [str(r["novel_outputs"][i]["out"]).replace("|", "\\|")
                 for r in results if r["novel_outputs"]]
        lines.append("| %s | %s | %s |" % (en, nl, " | ".join(cells)))
    lines.append("")

    out = out_dir / "SWEEP.md"
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
