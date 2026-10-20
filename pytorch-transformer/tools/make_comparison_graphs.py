"""Cross-run figures: WordLevel vs BPE.

Three figures, all built from artifacts that are already in the repo:

1. `val_bleu_vs_epochs.png`     - validation BLEU-4 per epoch, both runs
2. `overfitting_vs_epochs.png`  - the same, next to the training loss of each
                                  run: the moment the BLEU line turns down
                                  while the loss keeps falling is the moment
                                  the model starts memorising
3. `memorisation_evidence.png`  - teacher-forced loss on train vs val for both
                                  checkpoints (from overfit_diagnostic json)

Usage:
    python tools/make_comparison_graphs.py
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _paths import (RUNS_DIR, BPE_RUNS_DIR, BPE_WEIGHTS_DIR,  # noqa: E402
                    COMPARISON_GRAPHS_DIR, PROJECT_DIR)

STEPS_PER_EPOCH = 4349
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6f9f"     # blue  = WordLevel
ACCENT2 = "#c2622d"    # orange = BPE
GRID = "#e4e7eb"
RED = "#b3261e"


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


def step_to_epoch(step):
    if step <= 0:
        return 0
    return (step - 1) // STEPS_PER_EPOCH


def epoch_loss_from_events(runs_dir):
    """{epoch: mean training loss} from the per-step 'train loss' scalar."""
    per_epoch = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "events.out.tfevents.*"))):
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for e in ea.Scalars("train loss") if "train loss" in ea.Tags()["scalars"] else []:
            per_epoch.setdefault(step_to_epoch(e.step), []).append(e.value)
    return {k: sum(v) / len(v) for k, v in sorted(per_epoch.items())}


def events_scalars(runs_dir, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "events.out.tfevents.*"))):
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for e in ea.Scalars(tag) if tag in ea.Tags()["scalars"] else []:
            out.setdefault(step_to_epoch(e.step), e.value)
    return dict(sorted(out.items()))


def bpe_csv():
    """The BPE run's per-epoch log (complete, 20 rows)."""
    rows = {}
    path = BPE_WEIGHTS_DIR / "training_metrics.csv"
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["epoch"])] = {
                "train_loss": float(r["train_loss"]),
                "bleu4": float(r["bleu4"]),
                "cer": float(r["cer"]),
                "wer": float(r["wer"]),
            }
    return rows


def main():
    COMPARISON_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    bpe = bpe_csv()
    wl_loss = epoch_loss_from_events(RUNS_DIR)
    bpe_loss = epoch_loss_from_events(BPE_RUNS_DIR)
    wl_bleu = events_scalars(RUNS_DIR, "validation BLEU")

    # the WordLevel run only logged validation BLEU in its last epochs
    wl_bleu = {e: v for e, v in wl_bleu.items() if v > 0}
    bpe_bleu = {e: r["bleu4"] for e, r in bpe.items()}

    # ---- 1. validation BLEU per epoch ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sorted(bpe_bleu), [bpe_bleu[e] for e in sorted(bpe_bleu)], "o-",
            color=ACCENT2, linewidth=2, markersize=6, label="BPE 8000 (all 20 epochs logged)")
    ax.plot(sorted(wl_bleu), [wl_bleu[e] for e in sorted(wl_bleu)], "s--",
            color=ACCENT, linewidth=2.5, markersize=8, label="WordLevel (only the last 3 epochs were logged)")
    bpe_peak = max(bpe_bleu, key=lambda e: bpe_bleu[e])
    wl_peak = max(wl_bleu, key=lambda e: wl_bleu[e]) if wl_bleu else None
    ax.scatter([bpe_peak], [bpe_bleu[bpe_peak]], s=140, color=ACCENT2, zorder=5)
    ax.annotate("BPE peak: epoch %d, BLEU %.2f" % (bpe_peak, bpe_bleu[bpe_peak]),
                (bpe_peak, bpe_bleu[bpe_peak]), textcoords="offset points", xytext=(-120, 14),
                fontsize=10, color=ACCENT2, weight="bold")
    if wl_peak is not None:
        ax.scatter([wl_peak], [wl_bleu[wl_peak]], s=140, color=ACCENT, zorder=5)
        ax.annotate("WordLevel peak: epoch %d, BLEU %.2f" % (wl_peak, wl_bleu[wl_peak]),
                    (wl_peak, wl_bleu[wl_peak]), textcoords="offset points", xytext=(-160, 14),
                    fontsize=10, color=ACCENT, weight="bold")
    style(ax, "Validation BLEU-4 per epoch - WordLevel vs BPE", "epoch", "BLEU-4")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(COMPARISON_GRAPHS_DIR / "val_bleu_vs_epochs.png", dpi=150)
    plt.close(fig)
    print("wrote", COMPARISON_GRAPHS_DIR / "val_bleu_vs_epochs.png")

    # ---- 2. BLEU next to training loss ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    a = axes[0]
    a.plot(sorted(bpe_bleu), [bpe_bleu[e] for e in sorted(bpe_bleu)], "o-",
           color=ACCENT2, linewidth=2, markersize=5, label="BPE 8000")
    a.plot(sorted(wl_bleu), [wl_bleu[e] for e in sorted(wl_bleu)], "s--",
           color=ACCENT, linewidth=2.5, markersize=8, label="WordLevel (last 3 epochs only)")
    a.axvspan(bpe_peak, max(bpe_bleu), color="#fdecea", zorder=0)
    a.text(bpe_peak + 0.4, a.get_ylim()[0], "BPE is memorising", color=RED,
           fontsize=9, va="bottom", style="italic")
    style(a, "Validation BLEU-4", "epoch", "BLEU-4")
    a.legend(frameon=False, fontsize=9, loc="upper left")

    a = axes[1]
    e = sorted(bpe_loss)
    a.plot(e, [bpe_loss[k] for k in e], "o-", color=ACCENT2, linewidth=2,
           markersize=5, label="BPE 8000")
    e = sorted(wl_loss)
    a.plot(e, [wl_loss[k] for k in e], "s--", color=ACCENT, linewidth=2.5,
           markersize=6, label="WordLevel")
    style(a, "Training loss (epoch mean)", "epoch", "cross-entropy loss")
    a.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("Where each run starts overfitting", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(COMPARISON_GRAPHS_DIR / "overfitting_vs_epochs.png", dpi=150)
    plt.close(fig)
    print("wrote", COMPARISON_GRAPHS_DIR / "overfitting_vs_epochs.png")

    # ---- 3. memorisation evidence ---------------------------------------
    res = PROJECT_DIR / "results" / "comparison"
    data = {}
    for name, key in (("wordlevel", "WordLevel ep17"), ("bpe", "BPE ep17")):
        p = res / ("overfit_diagnostic_%s.json" % name)
        if p.exists():
            data[key] = json.loads(p.read_text())
    if len(data) == 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = list(data.keys())
        x = range(len(labels))
        w = 0.35
        tr = [data[l]["train_nats_per_token"] for l in labels]
        va = [data[l]["val_nats_per_token"] for l in labels]
        ax.bar([i - w / 2 for i in x], tr, w, color=ACCENT2, label="train loss (memorised data)")
        ax.bar([i + w / 2 for i in x], va, w, color=ACCENT, label="val loss (unseen data)")
        for i, (t, v) in enumerate(zip(tr, va)):
            ax.text(i - w / 2, t + 0.08, "%.2f" % t, ha="center", fontsize=10, color=INK)
            ax.text(i + w / 2, v + 0.08, "%.2f" % v, ha="center", fontsize=10, color=INK)
            ax.text(i, max(t, v) + 0.55, "gap %.2f" % (v - t), ha="center", fontsize=10,
                    color=RED, weight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(va) * 1.25)
        style(ax, "Memorisation: teacher-forced loss per token (lower is better)",
              "checkpoint", "nats per token")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(COMPARISON_GRAPHS_DIR / "memorisation_evidence.png", dpi=150)
        plt.close(fig)
        print("wrote", COMPARISON_GRAPHS_DIR / "memorisation_evidence.png")


if __name__ == "__main__":
    main()
