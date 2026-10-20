"""Evaluate checkpoints of any tokenizer variant with ONE identical protocol.

`evaluate.py` scores the checkpoint of the *current* variant (the folders come
from `config.get_config()`), and `compare_checkpoints.py` only compares
checkpoints of the same variant. Neither can put a WordLevel checkpoint and a
BPE checkpoint on the same table, because each variant has its own tokenizer
and its own vocabulary size.

This tool does: you hand it a list of `variant:path` specs and it decodes the
*same* seeded held-out sentences with every one of them, using that variant's
own tokenizer, and scores all of them the same way. That makes
"WordLevel epoch 17 vs BPE epoch 19" an apples-to-apples comparison.

Usage:
    python tools/eval_variants.py --specs wordlevel:model/tmodel_17.pt \
        bpe:drive_bpe/transformers-from-scratch_weights_bpe/tmodel_19.pt
    python tools/eval_variants.py --specs ... --samples 200 --no-novel

Output (under results/comparison/):
    eval_<label>.json     one metrics file per checkpoint
    compare_<labels>.md   the side-by-side table
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torchmetrics
from tokenizers import Tokenizer
from torch.utils.data import random_split
from tqdm import tqdm

from _paths import PROJECT_DIR, RESULTS_DIR  # noqa: F401  (must precede the project imports)
from config import get_config
from dataset import BilingualDataset
from model import build_transformer
from train import corpus_bleu, greedy_decode
from novel_sentences import SENTENCES

# Where the tokenizers live, per variant. The checkpoints are given by path,
# so the tokenizers are looked up next to the checkpoint (BPE) or in the
# project model folder (WordLevel).
TOKENIZER_DIRS = {
    "wordlevel": PROJECT_DIR / "model",
    "bpe": PROJECT_DIR / "drive_bpe" / "transformers-from-scratch_weights_bpe",
}


def parse_spec(spec):
    """'bpe:some/path.pt' -> ('bpe', Path('some/path.pt'))."""
    if ":" in spec:
        variant, path = spec.split(":", 1)
    else:
        variant, path = "wordlevel", spec
    variant = variant.strip().lower()
    if variant not in TOKENIZER_DIRS:
        raise SystemExit("unknown variant %r (expected one of %s)"
                         % (variant, ", ".join(TOKENIZER_DIRS)))
    return variant, Path(path.strip())


def build_val_dataset(config, tokenizer_src, tokenizer_tgt):
    """The exact seeded held-out split used during training."""
    from datasets import load_dataset
    ds_raw = load_dataset(config["datasource"],
                           f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    seed = config.get("seed", 42)
    train_size = int(0.9 * len(ds_raw))
    _, val_raw = random_split(ds_raw, [train_size, len(ds_raw) - train_size],
                              generator=torch.Generator().manual_seed(seed))
    return BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt,
                            config["lang_src"], config["lang_tgt"], config["seq_len"])


def load_model(config, variant, checkpoint, device):
    tok_dir = TOKENIZER_DIRS[variant]
    tokenizer_src = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_src']}.json"))
    tokenizer_tgt = Tokenizer.from_file(str(tok_dir / f"tokenizer_{config['lang_tgt']}.json"))
    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    return model, tokenizer_src, tokenizer_tgt, state, params_M


def decode_val(model, tokenizer_tgt, val_ds, n, config, device):
    """Greedy decode the first n held-out sentences."""
    src, ref, pred, gen_len = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(min(n, len(val_ds))), desc="decoding", leave=False):
            item = val_ds[i]
            out = greedy_decode(
                model,
                item["encoder_input"].unsqueeze(0).to(device),
                item["encoder_mask"].unsqueeze(0).to(device),
                None, tokenizer_tgt, config["seq_len"], device,
            )
            src.append(item["src_text"])
            ref.append(item["tgt_text"])
            pred.append(tokenizer_tgt.decode(out.detach().cpu().numpy()))
            gen_len.append(len(out))
    return src, ref, pred, gen_len


def decode_novel(model, tokenizer_src, tokenizer_tgt, config, device):
    """Translate the hand-written never-seen sentences."""
    seq_len = config["seq_len"]
    pad = tokenizer_src.token_to_id("[PAD]")
    out_texts = []
    with torch.no_grad():
        for en, _, _ in SENTENCES:
            ids = tokenizer_src.encode(en).ids[:seq_len - 2]
            source = torch.cat([
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor([pad] * (seq_len - len(ids) - 2), dtype=torch.int64),
            ]).unsqueeze(0).to(device)
            mask = (source != pad).unsqueeze(1).unsqueeze(1).int().to(device)
            out = greedy_decode(model, source, mask, None, tokenizer_tgt, seq_len, device)
            out_texts.append(tokenizer_tgt.decode(out.detach().cpu().numpy()))
    return out_texts


def _tok(text):
    """sacreBLEU 13a tokenization - the same one corpus_bleu uses."""
    import re
    text = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    text = text.replace("&amp;", "&")
    text = " " + text + " "
    text = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", text)
    text = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", text)
    text = re.sub(r"([\.,])([^0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])(-)", r"\1 \2 ", text)
    return text.split()


def score_block(pred, ref):
    bleu, prec, bp = corpus_bleu(pred, ref)
    cer = float(torchmetrics.CharErrorRate()(pred, ref))
    wer = float(torchmetrics.WordErrorRate()(pred, ref))
    exact = sum(1 for p, r in zip(pred, ref) if p.strip() == r.strip())
    return {
        "bleu4": round(bleu, 4),
        "precisions": [round(x, 4) for x in prec],
        "bp": round(bp, 4),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact": exact,
    }


def repeated_4gram_sentences(pred):
    """Sentences where some word 4-gram occurs more than once (a loop)."""
    n = 0
    for p in pred:
        toks = _tok(p)
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        if len(grams) != len(set(grams)):
            n += 1
    return n


def novel_hits(preds):
    """Never-seen sentences sharing >=50% of the reference words."""
    hits = 0
    for (en, nl, _), pred in zip(SENTENCES, preds):
        pset = {w.lower().strip(".,!?;:") for w in pred.split() if w.strip(".,!?;:")}
        rset = {w.lower().strip(".,!?;:") for w in nl.split() if w.strip(".,!?;:")}
        if pset and rset and len(pset & rset) / len(rset) >= 0.5:
            hits += 1
    return hits


def unk_rate(texts, tokenizer):
    unk = tokenizer.token_to_id("[UNK]")
    total = hits = 0
    for t in texts:
        ids = tokenizer.encode(t).ids
        total += len(ids)
        hits += sum(1 for i in ids if i == unk)
    return hits / total if total else 0.0


def evaluate(spec, config, device, n_samples, want_novel):
    variant, checkpoint = spec
    label = "%s_%s" % (variant, checkpoint.stem)
    print("\n=== %s ===" % label, flush=True)
    t0 = time.time()
    model, ts, tt, state, params_M = load_model(config, variant, checkpoint, device)
    print("  epoch %s  global_step %s  vocab %d/%d  params %.2fM  (load %.0fs)"
          % (state["epoch"], state["global_step"], ts.get_vocab_size(),
             tt.get_vocab_size(), params_M, time.time() - t0), flush=True)

    val_ds = build_val_dataset(config, ts, tt)
    src, ref, pred, gen_len = decode_val(model, tt, val_ds, n_samples, config, device)
    elapsed = time.time() - t0

    cased = score_block(pred, ref)
    case_blind = score_block([p.lower() for p in pred], [r.lower() for r in ref])

    novel_out = []
    novel_pred = []
    if want_novel:
        novel_pred = decode_novel(model, ts, tt, config, device)
        novel_out = [{"en": en, "ref": nl, "out": o}
                     for (en, nl, _), o in zip(SENTENCES, novel_pred)]

    result = {
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": state["epoch"],
        "global_step": state["global_step"],
        "params_M": round(params_M, 2),
        "vocab_src": ts.get_vocab_size(),
        "vocab_tgt": tt.get_vocab_size(),
        "n_sentences": len(pred),
        "cased": cased,
        "case_blind": case_blind,
        "mean_gen_len_tokens": round(statistics.fmean(gen_len), 2) if gen_len else 0.0,
        "mean_gen_len_words": round(statistics.fmean(len(p.split()) for p in pred), 2) if pred else 0.0,
        "mean_ref_len_words": round(statistics.fmean(len(r.split()) for r in ref), 2) if ref else 0.0,
        "mean_src_len_tokens": round(statistics.fmean(len(ts.encode(s).ids) for s in src), 2) if src else 0.0,
        "ref_unk_rate_tgt": round(unk_rate(ref, tt), 6),
        "src_unk_rate": round(unk_rate(src, ts), 6),
        "repeated_4gram_sentences": repeated_4gram_sentences(pred),
        "empty_predictions": sum(1 for p in pred if not p.strip()),
        "decode_seconds_total": round(elapsed, 1),
        "novel_outputs": novel_out,
        "novel_hits_ge50pct": novel_hits(novel_pred) if want_novel else None,
    }

    del model
    gc.collect()
    return label, result, (src, ref, pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", required=True,
                   help="variant:path entries, e.g. wordlevel:model/tmodel_17.pt")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--no-novel", action="store_true",
                   help="skip the 20 never-seen sentences")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    config = get_config()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device, "| samples:", args.samples, flush=True)

    specs = [parse_spec(s) for s in args.specs]
    for _, ck in specs:
        if not ck.exists():
            raise SystemExit("missing checkpoint: %s" % ck)

    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_DIR / "results" / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, results, decoded = [], [], []
    for spec in specs:
        label, res, dec = evaluate(spec, config, device, args.samples, not args.no_novel)
        labels.append(label)
        results.append(res)
        decoded.append(dec)
        (out_dir / ("eval_%s.json" % label)).write_text(json.dumps(res, indent=2))
        print("  cased BLEU %.2f | case-blind BLEU %.2f | CER %.3f | WER %.3f | novel %s/20"
              % (res["cased"]["bleu4"], res["case_blind"]["bleu4"],
                 res["cased"]["cer"], res["cased"]["wer"], res["novel_hits_ge50pct"]), flush=True)

    # --- comparison table --------------------------------------------------
    head = " | ".join(labels)
    lines = [
        "# Variant comparison - %s" % head,
        "",
        "- **Protocol:** greedy decoding, no beam search, %d identical held-out sentences"
        % args.samples,
        "- **Split:** seeded (seed %s) 90/10 split of `Helsinki-NLP/opus_books` (en-nl)"
        % config.get("seed"),
        "- **Case-blind** = prediction and reference lowercased before scoring "
        "(the BPE tokenizer lowercases its output, the WordLevel one does not)",
        "",
        "## Scores",
        "",
        "| Metric | " + head + " |",
        "|---|" + "---|" * len(labels),
    ]

    def row(name, getter):
        lines.append("| %s | " % name + " | ".join(str(getter(r)) for r in results) + " |")

    row("epoch", lambda r: r["checkpoint_epoch"])
    row("global_step", lambda r: r["global_step"])
    row("params (M)", lambda r: r["params_M"])
    row("vocab src / tgt", lambda r: "%d / %d" % (r["vocab_src"], r["vocab_tgt"]))
    row("BLEU-4 (cased)", lambda r: "%.2f" % r["cased"]["bleu4"])
    row("BLEU-4 (case-blind)", lambda r: "%.2f" % r["case_blind"]["bleu4"])
    row("BLEU-1 prec", lambda r: "%.4f" % r["case_blind"]["precisions"][0])
    row("BLEU-2 prec", lambda r: "%.4f" % r["case_blind"]["precisions"][1])
    row("BLEU-3 prec", lambda r: "%.4f" % r["case_blind"]["precisions"][2])
    row("BLEU-4 prec", lambda r: "%.4f" % r["case_blind"]["precisions"][3])
    row("brevity penalty", lambda r: "%.4f" % r["case_blind"]["bp"])
    row("CER", lambda r: "%.4f" % r["case_blind"]["cer"])
    row("WER", lambda r: "%.4f" % r["case_blind"]["wer"])
    row("exact match", lambda r: r["case_blind"]["exact"])
    row("mean gen len (tokens)", lambda r: r["mean_gen_len_tokens"])
    row("mean gen len (words)", lambda r: r["mean_gen_len_words"])
    row("src [UNK] rate", lambda r: "%.4f" % r["src_unk_rate"])
    row("ref [UNK] rate", lambda r: "%.4f" % r["ref_unk_rate_tgt"])
    row("looped sentences (4-gram)", lambda r: r["repeated_4gram_sentences"])
    row("empty predictions", lambda r: r["empty_predictions"])
    row("never-seen hits >=50% ref", lambda r: r["novel_hits_ge50pct"])
    row("decode seconds", lambda r: r["decode_seconds_total"])

    best = max(results, key=lambda r: r["case_blind"]["bleu4"])
    lines += [
        "",
        "## Verdict",
        "",
        "**Best on case-blind BLEU-4: `%s` (epoch %s) - %.2f**"
        % (labels[results.index(best)], best["checkpoint_epoch"], best["case_blind"]["bleu4"]),
        "",
    ]

    # per-sentence head-to-head
    if len(results) > 1:
        lines += ["## Head-to-head per sentence (case-blind BLEU)", "",
                  "| checkpoint | sentences won | best on |", "|---|---|---|"]
        per = []
        for (src, ref, pred) in decoded:
            per.append([corpus_bleu([a.lower()], [b.lower()])[0] for a, b in zip(pred, ref)])
        wins = {lab: 0 for lab in labels}
        ties = 0
        for row_ in zip(*per):
            best_v = max(row_)
            if row_.count(best_v) > 1:
                ties += 1
            else:
                wins[labels[row_.index(best_v)]] += 1
        for lab in labels:
            lines.append("| `%s` | %d | %s |" % (lab, wins[lab], labels[max(
                range(len(labels)), key=lambda j: sum(1 for v in per[j] if v == max(row_)))]))
        lines += ["| tied | %d | |" % ties, ""]

    # novel sentence table
    if not args.no_novel:
        lines += ["## Never-seen sentences", "",
                  "| English | Reference | " + " | ".join("`%s`" % l for l in labels) + " |",
                  "|---|---|" + "---|" * len(labels)]
        for i, (en, nl, _) in enumerate(SENTENCES):
            cells = [str(r["novel_outputs"][i]["out"]).replace("|", "\\|") for r in results]
            lines.append("| %s | %s | %s |" % (en, nl, " | ".join(cells)))
        lines.append("")

    out = out_dir / ("compare_%s.md" % "_vs_".join(labels))
    out.write_text("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
