"""Compare two or more checkpoints on the same sentences.

Decodes an identical set of held-out sentences with every checkpoint, scores
each, and writes a side-by-side markdown table plus a metric comparison graph.
Takes the best checkpoint per metric and declares an overall winner.

Usage:
    python compare_checkpoints.py --checkpoints tmodel_17.pt tmodel_18.pt tmodel_19.pt
    python compare_checkpoints.py --samples 60 --num-print 20
"""

import argparse
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchmetrics
from tokenizers import Tokenizer
from tqdm import tqdm

from _paths import RESULTS_DIR, GRAPHS_DIR  # noqa: F401  (must precede the project imports)
from config import get_config, resolve_checkpoint
from model import build_transformer
from train import corpus_bleu, get_ds, greedy_decode

# metric key -> (label, higher_is_better)
METRICS = [
    ("bleu4", "BLEU-4", True),
    ("p1", "BLEU-1 prec", True),
    ("p2", "BLEU-2 prec", True),
    ("p3", "BLEU-3 prec", True),
    ("p4", "BLEU-4 prec", True),
    ("bp", "brevity pen", True),
    ("cer", "CER", False),
    ("wer", "WER", False),
    ("mean_sentence_bleu", "mean sent BLEU", True),
    ("median_sentence_bleu", "median sent BLEU", True),
    ("nonzero_bleu_sents", "sents with BLEU>0", True),
    ("exact", "exact match", True),
    ("mean_gen_len", "mean gen len", None),
]

COLORS = ["#2f6f9f", "#c2622d", "#3f7d58", "#8e44ad", "#b03a48"]


def build(config, checkpoint, device):
    ts = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))
    model = build_transformer(
        ts.get_vocab_size(), tt.get_vocab_size(),
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, tt, state


def decode_all(model, tt, loader, n, config, device):
    src, exp, pred = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="decoding", total=n, leave=False)):
            if i >= n:
                break
            out = greedy_decode(
                model, batch["encoder_input"].to(device), batch["encoder_mask"].to(device),
                None, tt, config["seq_len"], device,
            )
            src.append(batch["src_text"][0])
            exp.append(batch["tgt_text"][0])
            pred.append(tt.decode(out.detach().cpu().numpy()))
    return src, exp, pred


def score(pred, exp, gen_len):
    bleu, prec, bp = corpus_bleu(pred, exp)
    per = [corpus_bleu([a], [b])[0] for a, b in zip(pred, exp)]
    return {
        "bleu4": round(bleu, 3),
        "p1": round(prec[0], 4), "p2": round(prec[1], 4),
        "p3": round(prec[2], 4), "p4": round(prec[3], 4),
        "bp": round(bp, 4),
        "cer": round(float(torchmetrics.CharErrorRate()(pred, exp)), 4),
        "wer": round(float(torchmetrics.WordErrorRate()(pred, exp)), 4),
        "exact": sum(1 for a, b in zip(pred, exp) if a.strip() == b.strip()),
        "mean_sentence_bleu": round(statistics.fmean(per) if per else 0.0, 3),
        "median_sentence_bleu": round(statistics.median(per) if per else 0.0, 3),
        "nonzero_bleu_sents": sum(1 for x in per if x > 0),
        "mean_gen_len": round(statistics.fmean(gen_len) if gen_len else 0.0, 2),
        "empty": sum(1 for x in pred if not x.strip()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="epoch numbers or names, e.g. 17 18 19 (default: 17 18 19)")
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--num-print", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    config = get_config()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)

    specs = args.checkpoints or ["tmodel_17.pt", "tmodel_18.pt", "tmodel_19.pt"]
    cks = [resolve_checkpoint(config, s) for s in specs]
    for c in cks:
        if not Path(c).exists():
            raise SystemExit(f"missing checkpoint: {c}")

    _, val_loader, _, _ = get_ds(config)
    n = min(args.samples, len(val_loader.dataset))
    print(f"decoding {n} identical sentences with each of {len(cks)} checkpoints\n")

    runs = []
    for ck in cks:
        t0 = time.time()
        model, tt, state = build(config, ck, device)
        src, exp, pred = decode_all(model, tt, val_loader, n, config, device)
        gen_len = [len(p.split()) for p in pred]
        m = score(pred, exp, gen_len)
        m["checkpoint"] = Path(ck).stem
        m["epoch"] = state["epoch"]
        m["global_step"] = state["global_step"]
        m["seconds"] = round(time.time() - t0, 1)
        m["per_sentence_bleu"] = [corpus_bleu([a], [b])[0] for a, b in zip(pred, exp)]
        runs.append({"meta": m, "src": src, "exp": exp, "pred": pred})
        print("%-14s ep%-3d BLEU %6.2f  CER %.4f  WER %.4f  exact %d  (%.0fs)"
              % (m["checkpoint"], m["epoch"], m["bleu4"], m["cer"], m["wer"],
                 m["exact"], m["seconds"]))

    names = [r["meta"]["checkpoint"] for r in runs]

    # --- overall winner: wins on the metrics that have a clear direction -----
    print("\n--- per-metric winners ---")
    wins = {nm: 0 for nm in names}
    for key, label, higher in METRICS:
        if higher is None:
            continue
        vals = [r["meta"][key] for r in runs]
        best = max(vals) if higher else min(vals)
        w = names[vals.index(best)]
        wins[w] += 1
        print("  %-18s %s" % (label, w))
    overall = max(wins, key=lambda k: wins[k])
    print("\noverall winner: %s (%d of %d directed metrics)"
          % (overall, wins[overall], sum(1 for _, _, h in METRICS if h is not None)))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # --- markdown ----------------------------------------------------------
    head = " vs ".join(names)
    lines = [
        "# Checkpoint comparison - %s" % head,
        "",
        "- **Scored on:** %d identical held-out sentences (seed %s)" % (n, config.get("seed")),
        "- **Decoding:** greedy | **device:** %s" % device,
        "",
        "## Scores",
        "",
        "| Metric | " + " | ".join("`%s` (ep %d)" % (r["meta"]["checkpoint"], r["meta"]["epoch"])
                                    for r in runs) + " | Best |",
        "|---|" + "---|" * (len(runs) + 1),
    ]
    for key, label, higher in METRICS:
        vals = [r["meta"][key] for r in runs]
        if higher is None:
            best = "-"
        else:
            b = max(vals) if higher else min(vals)
            best = "`%s`" % names[vals.index(b)]
        cells = " | ".join("%.4f" % v if isinstance(v, float) else str(v) for v in vals)
        lines.append("| %s | %s | %s |" % (label, cells, best))

    lines += ["", "## Overall", "", "Wins per directed metric: " +
              ", ".join("`%s` %d" % (k, v) for k, v in wins.items()), "",
              "**Winner: `%s`**" % overall, ""]

    if len(runs) > 1:
        # per-sentence BLEU, one column per checkpoint
        per = [r["meta"]["per_sentence_bleu"] for r in runs]
        best_on = [max(range(len(names)), key=lambda j: vals[j]) for vals in zip(*per)]
        n_best = [sum(1 for v in vals if v == max(vals)) for vals in zip(*per)]
        ties = sum(1 for k in n_best if k > 1)
        lines += ["## Head-to-head per sentence", "",
                  "Sentences where each checkpoint had the best per-sentence BLEU:", "",
                  "| | count |", "|---|---|"]
        for j, nm in enumerate(names):
            lines.append("| `%s` best | %d |" % (nm, best_on.count(j)))
        lines += ["| tied across checkpoints | %d |" % ties, ""]

    lines += ["## Side-by-side translations", ""]
    # spread of per-sentence BLEU across the checkpoints, so the sentences the
    # checkpoints disagree most about come first
    diffs = [max(v) - min(v) for v in
             zip(*[r["meta"]["per_sentence_bleu"] for r in runs])]
    order = sorted(range(len(diffs)), key=lambda i: -diffs[i]) if len(runs) > 1 else range(len(diffs))
    for k, i in enumerate(order[: args.num_print], start=1):
        lines += ["### %d" % k, "", "- **SOURCE:** %s" % runs[0]["src"][i],
                  "- **REFERENCE:** %s" % runs[0]["exp"][i]]
        for r in runs:
            lines.append("- **`%s`:** %s" % (r["meta"]["checkpoint"], r["pred"][i] or "*(empty)*"))
        lines.append("")

    out = RESULTS_DIR / ("compare_%s.md" % "_vs_".join(names))
    out.write_text("\n".join(lines))
    print("\nwrote", out)

    # --- graph -------------------------------------------------------------
    xlabels = ["BLEU-4", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CER", "WER"]
    keys = ["bleu4", "p1", "p2", "p3", "p4", "cer", "wer"]
    scale = [1, 100, 100, 100, 100, 100, 100]  # precisions and error rates as %
    x = range(len(xlabels))
    w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for j, r in enumerate(runs):
        vals = [r["meta"][k] * s for k, s in zip(keys, scale)]
        ax.bar([i + (j - (len(runs) - 1) / 2) * w for i in x], vals, w,
               label="%s (epoch %d)" % (r["meta"]["checkpoint"], r["meta"]["epoch"]),
               color=COLORS[j % len(COLORS)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("BLEU / precision in %   (CER & WER also in %)")
    ax.set_title("Checkpoint comparison on %d held-out sentences\nlower is better for CER and WER, higher for the rest"
                 % n, fontsize=11, loc="left", color="#1f2933")
    ax.grid(True, axis="y", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "checkpoint_comparison.png", dpi=150)
    plt.close(fig)
    print("wrote", GRAPHS_DIR / "checkpoint_comparison.png")


if __name__ == "__main__":
    main()
