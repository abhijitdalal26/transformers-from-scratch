"""Regenerate every training graph from the TensorBoard event files.

The event files live in drive/transformers-from-scratch_runs/tmodel_en_nl/ and are
the raw training log. This script reads all of them, merges the scalars and writes
a clean set of PNGs into graphs/.

Note: the log has two recording gaps (the run was restarted a few times and some
event files were never uploaded). Those regions are shaded so the curves are not
misleading.

Usage:
    python tools/make_graphs.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _paths import RUNS_DIR, GRAPHS_DIR  # noqa: E402  (must precede the project imports)

# graphs are split by tokenizer so a future BPE run cannot overwrite the
# WordLevel baseline: graphs/wordlevel/ vs graphs/bpe/
STEPS_PER_EPOCH = 4349
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6f9f"
ACCENT2 = "#c2622d"
GRID = "#e4e7eb"


def load_scalars():
    """Merge every event file into {tag: {step: value}} (first write wins)."""
    acc = {}
    for f in sorted(glob.glob(os.path.join(RUNS_DIR, "events.out.tfevents.*"))):
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                acc.setdefault(tag, {}).setdefault(e.step, e.value)
    return acc


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
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    acc = load_scalars()
    print("tags found:", ", ".join(sorted(acc)))

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

        sm = [statistics.fmean(loss_y[max(0, i - w):i + 1]) for i in range(len(loss_y))]
        ax.plot(loss_x, sm, color=ACCENT2, linewidth=2.0, label="500-step mean")
    shade_gaps(ax, gaps, label=True)
    ax.set_yscale("log")
    style(ax, "Training loss  (tmodel, opus_books en->nl, 20 epochs)",
          "global step", "cross-entropy loss (log scale)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "train_loss.png", dpi=150)
    plt.close(fig)
    print("wrote", GRAPHS_DIR / "train_loss.png")

    # ---- 2. validation BLEU ---------------------------------------------
    bx, by = series(acc, "validation BLEU")
    # the first 10 epochs were logged with a broken BLEU that always returned 0
    real = [(x, y) for x, y in zip(bx, by) if y > 0]
    if real:
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [x for x, _ in real]
        ys = [y for _, y in real]
        ax.plot([x / STEPS_PER_EPOCH for x in xs], ys, "o-",
                color=ACCENT, linewidth=2, markersize=7, label="BLEU-4 (200 sentences)")
        best = max(zip(xs, ys), key=lambda t: t[1])
        ax.scatter([best[0] / STEPS_PER_EPOCH], [best[1]], s=140, color=ACCENT2,
                   zorder=5, label="best: epoch %d, BLEU %.2f" % (best[0] // STEPS_PER_EPOCH, best[1]))
        ax.annotate("%.2f" % best[1], (best[0] / STEPS_PER_EPOCH, best[1]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, color=ACCENT2, weight="bold")
        style(ax, "Validation BLEU-4 per epoch", "epoch", "BLEU-4")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / "val_bleu.png", dpi=150)
        plt.close(fig)
        print("wrote", GRAPHS_DIR / "val_bleu.png")

    # ---- 3. CER / WER ----------------------------------------------------
    cx, cy = series(acc, "validation cer")
    wx, wy = series(acc, "validation wer")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([x / STEPS_PER_EPOCH for x in cx], cy, "o-", color=ACCENT, linewidth=2,
            markersize=6, label="CER (character error rate)")
    ax.plot([x / STEPS_PER_EPOCH for x in wx], wy, "s-", color=ACCENT2, linewidth=2,
            markersize=6, label="WER (word error rate)")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.02, 1.01, "WER > 1.0 = more wrong words than reference words",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    style(ax, "Validation error rates per epoch", "epoch", "error rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "val_error_rates.png", dpi=150)
    plt.close(fig)
    print("wrote", GRAPHS_DIR / "val_error_rates.png")

    # ---- 4. BLEU n-gram precisions --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in (1, 2, 3, 4):
        px, py = series(acc, "validation BLEU p%d" % i)
        if not px:
            continue
        ax.plot([x / STEPS_PER_EPOCH for x in px], [v * 100 for v in py], "o-",
                linewidth=2, markersize=6, label="BLEU-%d precision" % i)
    style(ax, "BLEU n-gram precisions (last 3 epochs only)", "epoch", "precision %")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "bleu_precisions.png", dpi=150)
    plt.close(fig)
    print("wrote", GRAPHS_DIR / "bleu_precisions.png")

    # ---- 5. combined dashboard -----------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("tmodel - English to Dutch Transformer (WordLevel), training summary",
                 fontsize=14, color=INK, x=0.02, ha="left")

    a = axes[0][0]
    a.plot(loss_x, loss_y, color=ACCENT, linewidth=0.5, alpha=0.5)
    shade_gaps(a, gaps)
    a.set_yscale("log")
    style(a, "Training loss", "global step", "loss")

    a = axes[0][1]
    if real:
        a.plot([x / STEPS_PER_EPOCH for x, _ in real], [y for _, y in real], "o-",
               color=ACCENT, linewidth=2, markersize=6)
        style(a, "Validation BLEU-4", "epoch", "BLEU-4")

    a = axes[1][0]
    a.plot([x / STEPS_PER_EPOCH for x in cx], cy, "o-", color=ACCENT, linewidth=2,
           markersize=6, label="CER")
    a.plot([x / STEPS_PER_EPOCH for x in wx], wy, "s-", color=ACCENT2, linewidth=2,
           markersize=6, label="WER")
    style(a, "Validation error rates", "epoch", "error rate")
    a.legend(frameon=False, fontsize=9)

    a = axes[1][1]
    for i, c in zip((1, 2, 3, 4), (ACCENT, ACCENT2, "#3f7d58", "#8e44ad")):
        px, py = series(acc, "validation BLEU p%d" % i)
        if px:
            a.plot([x / STEPS_PER_EPOCH for x in px], [v * 100 for v in py], "o-",
                   color=c, linewidth=2, markersize=6, label="p%d" % i)
    style(a, "BLEU n-gram precisions", "epoch", "precision %")
    a.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(GRAPHS_DIR / "training_dashboard.png", dpi=150)
    plt.close(fig)
    print("wrote", GRAPHS_DIR / "training_dashboard.png")


if __name__ == "__main__":
    main()
