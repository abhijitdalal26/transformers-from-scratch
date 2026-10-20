"""Regenerate every training graph from the TensorBoard event files.

The event files live in drive/transformers-from-scratch_runs/tmodel_en_nl/ and are
the raw training log. This script reads all of them, merges the scalars and writes
a clean set of PNGs into graphs/.

Note: the log has two recording gaps (the run was restarted a few times and some
event files were never uploaded). Those regions are shaded so the curves are not
misleading.

Usage:
    python tools/make_graphs.py                  # WordLevel baseline
    python tools/make_graphs.py --variant bpe    # the BPE run
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _paths import RUNS_DIR, GRAPHS_DIR, BPE_RUNS_DIR, BPE_GRAPHS_DIR, BPE_WEIGHTS_DIR  # noqa: E402  (must precede the project imports)

# graphs are split by tokenizer so a future BPE run cannot overwrite the
# WordLevel baseline: graphs/wordlevel/ vs graphs/bpe/
STEPS_PER_EPOCH = 4349
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6f9f"
ACCENT2 = "#c2622d"
GRID = "#e4e7eb"

# per-variant settings: where the event files are, where the PNGs go, how the
# plots are titled
VARIANTS = {
    "wordlevel": {
        "runs": RUNS_DIR,
        "graphs": GRAPHS_DIR,
        "label": "WordLevel",
        "title": "tmodel - English to Dutch Transformer (WordLevel), training summary",
    },
    "bpe": {
        "runs": BPE_RUNS_DIR,
        "graphs": BPE_GRAPHS_DIR,
        "csv": BPE_WEIGHTS_DIR / "training_metrics.csv",
        "label": "BPE (8000)",
        "title": "tmodel - English to Dutch Transformer (BPE 8000), training summary",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=sorted(VARIANTS), default="wordlevel")
    return p.parse_args()


def step_to_epoch(step):
    """global step -> epoch index, 0-based, matching training_metrics.csv."""
    if step <= 0:
        return 0
    return (step - 1) // STEPS_PER_EPOCH


def load_scalars(runs_dir):
    """Merge every event file into {tag: {step: value}} (first write wins)."""
    acc = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "events.out.tfevents.*"))):
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                acc.setdefault(tag, {}).setdefault(e.step, e.value)
    return acc


def load_metrics_csv(csv_path):
    """Per-epoch validation numbers from training_metrics.csv, keyed by step.

    The TensorBoard log of the BPE run is missing a couple of validation
    epochs (the run was restarted); the CSV has every epoch, so it wins.
    """
    import csv as _csv

    if not csv_path or not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            if not row.get("global_step"):
                continue
            step = int(row["global_step"])
            for tag, col in (("validation BLEU", "bleu4"), ("validation BLEU bp", "bp"),
                             ("validation BLEU p1", "p1"), ("validation BLEU p2", "p2"),
                             ("validation BLEU p3", "p3"), ("validation BLEU p4", "p4"),
                             ("validation cer", "cer"), ("validation wer", "wer")):
                val = row.get(col)
                if val not in (None, ""):
                    out.setdefault(tag, {})[step] = float(val)
    return out


def find_gaps(steps, min_gap=2000):
    gaps = []
    for a, b in zip(steps, steps[1:]):
        if b - a > min_gap:
            gaps.append((a, b))
    return gaps


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def shade_gaps(ax, gaps, label=False):
    for a, b in gaps:
        ax.axvspan(a, b, color="#f0f2f5", zorder=0)
    if label and gaps:
        ax.axvspan(*gaps[0], color="#f0f2f5", zorder=0, label="log gap (run restarted)")


def series(acc, tag):
    d = acc.get(tag, {})
    xs = sorted(d)
    return xs, [d[x] for x in xs]


def main():
    args = parse_args()
    v = VARIANTS[args.variant]
    runs_dir = str(v["runs"])
    graphs_dir = v["graphs"]
    label = v["label"]
    graphs_dir.mkdir(parents=True, exist_ok=True)
    acc = load_scalars(runs_dir)
    print("tags found:", ", ".join(sorted(acc)))

    # the per-epoch CSV is authoritative when it exists (complete, no gaps)
    csv_acc = load_metrics_csv(v.get("csv"))
    for tag, values in csv_acc.items():
        merged = dict(acc.get(tag, {}))
        merged.update(values)
        acc[tag] = merged
    if csv_acc:
        print("validation series taken from the metrics csv: %s" % ", ".join(sorted(csv_acc)))

    # ---- 1. training loss ------------------------------------------------
    loss_x, loss_y = series(acc, "train loss")
    gaps = find_gaps(loss_x)
    print("train loss points: %d, gaps: %s" % (len(loss_x), gaps))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(loss_x, loss_y, color=ACCENT, linewidth=0.5, alpha=0.55, label="per-step loss")
    # rolling mean to make the trend readable
    w = 500
    if len(loss_y) > w:
        import statistics

        sm = [statistics.fmean(loss_y[max(0, i - w):i + 1]) for i in range(len(loss_x))]
        ax.plot(loss_x, sm, color=ACCENT2, linewidth=2.0, label="500-step mean")
    shade_gaps(ax, gaps, label=True)
    ax.set_yscale("log")
    style(ax, "Training loss  (tmodel %s, opus_books en->nl, 20 epochs)" % label,
          "global step", "cross-entropy loss (log scale)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(graphs_dir / "train_loss.png", dpi=150)
    plt.close(fig)
    print("wrote", graphs_dir / "train_loss.png")

    # ---- 2. validation BLEU ---------------------------------------------
    bx, by = series(acc, "validation BLEU")
    # the first 10 epochs were logged with a broken BLEU that always returned 0
    real = [(x, y) for x, y in zip(bx, by) if y > 0]
    if real:
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [x for x, _ in real]
        ys = [y for _, y in real]
        ax.plot([step_to_epoch(x) for x in xs], ys, "o-",
                color=ACCENT, linewidth=2, markersize=7, label="BLEU-4 (200 sentences)")
        best = max(zip(xs, ys), key=lambda t: t[1])
        ax.scatter([step_to_epoch(best[0])], [best[1]], s=140, color=ACCENT2,
                   zorder=5, label="best: epoch %d, BLEU %.2f" % (step_to_epoch(best[0]), best[1]))
        ax.annotate("%.2f" % best[1], (step_to_epoch(best[0]), best[1]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, color=ACCENT2, weight="bold")
        style(ax, "Validation BLEU-4 per epoch  (%s)" % label, "epoch", "BLEU-4")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        fig.tight_layout()
        fig.savefig(graphs_dir / "val_bleu.png", dpi=150)
        plt.close(fig)
        print("wrote", graphs_dir / "val_bleu.png")

    # ---- 3. CER / WER ----------------------------------------------------
    cx, cy = series(acc, "validation cer")
    wx, wy = series(acc, "validation wer")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([step_to_epoch(x) for x in cx], cy, "o-", color=ACCENT, linewidth=2,
            markersize=6, label="CER (character error rate)")
    ax.plot([step_to_epoch(x) for x in wx], wy, "s-", color=ACCENT2, linewidth=2,
            markersize=6, label="WER (word error rate)")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.02, 1.01, "WER > 1.0 = more wrong words than reference words",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    style(ax, "Validation error rates per epoch  (%s)" % label, "epoch", "error rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(graphs_dir / "val_error_rates.png", dpi=150)
    plt.close(fig)
    print("wrote", graphs_dir / "val_error_rates.png")

    # ---- 4. BLEU n-gram precisions --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in (1, 2, 3, 4):
        px, py = series(acc, "validation BLEU p%d" % i)
        if not px:
            continue
        ax.plot([step_to_epoch(x) for x in px], [v * 100 for v in py], "o-",
                linewidth=2, markersize=6, label="BLEU-%d precision" % i)
    style(ax, "BLEU n-gram precisions", "epoch", "precision %")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(graphs_dir / "bleu_precisions.png", dpi=150)
    plt.close(fig)
    print("wrote", graphs_dir / "bleu_precisions.png")

    # ---- 5. combined dashboard -----------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(v["title"], fontsize=14, color=INK, x=0.02, ha="left")

    a = axes[0][0]
    a.plot(loss_x, loss_y, color=ACCENT, linewidth=0.5, alpha=0.5)
    shade_gaps(a, gaps)
    a.set_yscale("log")
    style(a, "Training loss", "global step", "loss")

    a = axes[0][1]
    if real:
        a.plot([step_to_epoch(x) for x, _ in real], [y for _, y in real], "o-",
               color=ACCENT, linewidth=2, markersize=6)
        style(a, "Validation BLEU-4", "epoch", "BLEU-4")

    a = axes[1][0]
    a.plot([step_to_epoch(x) for x in cx], cy, "o-", color=ACCENT, linewidth=2,
           markersize=6, label="CER")
    a.plot([step_to_epoch(x) for x in wx], wy, "s-", color=ACCENT2, linewidth=2,
           markersize=6, label="WER")
    style(a, "Validation error rates", "epoch", "error rate")
    a.legend(frameon=False, fontsize=9)

    a = axes[1][1]
    for i, c in zip((1, 2, 3, 4), (ACCENT, ACCENT2, "#3f7d58", "#8e44ad")):
        px, py = series(acc, "validation BLEU p%d" % i)
        if px:
            a.plot([step_to_epoch(x) for x in px], [v * 100 for v in py], "o-",
                   color=c, linewidth=2, markersize=6, label="p%d" % i)
    style(a, "BLEU n-gram precisions", "epoch", "precision %")
    a.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(graphs_dir / "training_dashboard.png", dpi=150)
    plt.close(fig)
    print("wrote", graphs_dir / "training_dashboard.png")

    # ---- 6. the overfitting chart --------------------------------------
    # train loss (per epoch mean) and validation BLEU on two axes: the moment
    # the blue line turns up while the orange line keeps falling is the moment
    # the model starts memorising instead of learning.
    if real:
        per_epoch_loss = {}
        for x, y in zip(loss_x, loss_y):
            per_epoch_loss.setdefault(step_to_epoch(x), []).append(y)
        epochs = sorted(per_epoch_loss)
        losses = [sum(per_epoch_loss[e]) / len(per_epoch_loss[e]) for e in epochs]
        bleus = {step_to_epoch(x): y for x, y in real}

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax2 = ax.twinx()
        be = [e for e in epochs if e in bleus]
        l1, = ax.plot(be, [bleus[e] for e in be], "o-",
                      color=ACCENT, linewidth=2, markersize=7, label="validation BLEU-4")
        l2, = ax2.plot(epochs, losses, "s--", color=ACCENT2, linewidth=2,
                       markersize=6, label="training loss (epoch mean)")
        best_ep = max(be, key=lambda e: bleus[e])
        ax.axvline(best_ep, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_ylim(top=max(bleus.values()) * 1.18)
        ax.annotate("peak: epoch %d\nBLEU %.2f" % (best_ep, bleus[best_ep]),
                    (best_ep, bleus[best_ep]), textcoords="offset points",
                    xytext=(10, 18), fontsize=10, color=INK, weight="bold")
        ax.axvspan(best_ep, max(epochs), color="#fdecea", zorder=0)
        ax.text(max(epochs) - 0.3, ax.get_ylim()[0], "memorising", color="#b3261e",
                fontsize=10, ha="right", va="bottom", style="italic")
        style(ax, "Overfitting: validation BLEU vs training loss  (%s)" % label,
              "epoch", "validation BLEU-4")
        style(ax2, "", "", "training loss")
        ax2.grid(False)
        ax.legend(handles=[l1, l2], frameon=False, fontsize=9, loc="lower left")
        fig.tight_layout()
        fig.savefig(graphs_dir / "overfitting.png", dpi=150)
        plt.close(fig)
        print("wrote", graphs_dir / "overfitting.png")


if __name__ == "__main__":
    main()
