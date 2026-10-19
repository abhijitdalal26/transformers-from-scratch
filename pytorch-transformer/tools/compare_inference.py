"""Side-by-side inference: the same sentences through several checkpoints.

`compare_checkpoints.py` answers "which checkpoint scores best?" on the
validation split. This answers the qualitative question instead - "for these
particular sentences, what does each checkpoint actually say, and which one
would you rather ship?" - using the never-seen sentences from
`novel_sentences.py`, so the reference Dutch is independent of the corpus.

Writes results/wordlevel/inference_<a>_vs_<b>.md and prints the same table.

Usage:
    python tools/compare_inference.py
    python tools/compare_inference.py --checkpoints model/tmodel_17.pt drive/tmodel_19.pt
    python tools/compare_inference.py --num-print 12
"""

import argparse
import contextlib
import io
from pathlib import Path

from _paths import PROJECT_DIR, RESULTS_DIR  # noqa: F401  (must precede the project imports)
from config import get_config, resolve_checkpoint
from novel_sentences import SENTENCES
from train import corpus_bleu
from translate import translate

# the validated best checkpoint vs the last one, i.e. the overfitting boundary
DEFAULT = [str(PROJECT_DIR / "model" / "tmodel_17.pt"),
           str(PROJECT_DIR / "drive" / "tmodel_19.pt")]


def decode_all(config, checkpoint):
    """Translate every sentence with one checkpoint.

    Checkpoint-outer, sentence-inner: translate() caches a single loaded model,
    so alternating between checkpoints would reload 838 MB on every sentence.
    """
    out = []
    for en, _, _ in SENTENCES:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out.append((translate(en, config, checkpoint=checkpoint) or "").strip())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="paths or epoch numbers (default: %s)" % " ".join(DEFAULT))
    p.add_argument("--num-print", type=int, default=0,
                   help="how many sentences to write (default: all %d)" % len(SENTENCES))
    args = p.parse_args()

    config = get_config()
    specs = args.checkpoints or DEFAULT
    cks = [resolve_checkpoint(config, s) for s in specs]
    for c in cks:
        if not Path(c).exists():
            raise SystemExit(f"missing checkpoint: {c}")
    names = [Path(c).stem for c in cks]

    outputs = []
    for ck, name in zip(cks, names):
        print("decoding %d sentences with %s ..." % (len(SENTENCES), name), flush=True)
        outputs.append(decode_all(config, ck))

    # Per-sentence BLEU against the hand-written Dutch reference. Weak on its
    # own - the model speaks 19th-century literary Dutch, the references are
    # modern - but it is the same yardstick for every checkpoint, so the
    # comparison between them is fair.
    per_ck = [[corpus_bleu([out], [ref])[0] for out, (_, ref, _) in zip(o, SENTENCES)]
              for o in outputs]
    # A strict win needs a higher score than *every* other checkpoint; a tie is
    # counted separately so the tally cannot be won by argument order.
    wins = {n: 0 for n in names}
    ties = 0
    for row in zip(*per_ck):
        best = max(row)
        if row.count(best) > 1:
            ties += 1
        else:
            wins[names[row.index(best)]] += 1

    # --- console ---------------------------------------------------------
    w_en = max(len(en) for en, _, _ in SENTENCES)
    for i, (en, ref, probe) in enumerate(SENTENCES):
        print("-" * (w_en + 20 * len(names) + 30))
        print("EN  %s" % en.ljust(w_en))
        print("REF %s" % ref)
        for j, name in enumerate(names):
            print("    %-12s %s" % (name, outputs[j][i] or "*(empty)*"))
        print("    per-sentence BLEU: " + "  ".join(
            "%s %.1f" % (n, per_ck[j][i]) for j, n in enumerate(names)))
        print("    [%s]" % probe)

    print("=" * 60)
    for j, name in enumerate(names):
        print("%-14s mean per-sentence BLEU %.2f   best on %d/%d sentences"
              % (name, sum(per_ck[j]) / len(per_ck[j]), wins[name], len(SENTENCES)))
    print("winner: %s" % max(wins, key=lambda k: wins[k]))

    # --- markdown --------------------------------------------------------
    lines = [
        "# Side-by-side inference - %s" % " vs ".join(names),
        "",
        "The same %d sentences translated by each checkpoint with greedy decoding."
        % len(SENTENCES),
        "",
        "These are the never-seen sentences from `novel_sentences.py`: written by hand,",
        "appearing in neither the training nor the validation split. Per-sentence BLEU",
        "is measured against the hand-written Dutch reference. The model was trained on",
        "19th-century literary prose and answers in that register, so the absolute BLEU",
        "is low against modern references - the point here is that every checkpoint is",
        "judged on the identical yardstick.",
        "",
        "| English | Reference | " + " | ".join("`%s`" % n for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for i, (en, ref, _) in enumerate(SENTENCES):
        lines.append("| %s | %s | %s |" % (
            en, ref, " | ".join((outputs[j][i] or "*(empty)*").replace("|", "\\|")
                                for j in range(len(names)))))

    lines += ["", "## Per-sentence BLEU against the reference", "",
              "| English | " + " | ".join("`%s`" % n for n in names) + " | better |",
              "|---|" + "---|" * (len(names) + 1)]
    for i, (en, _, _) in enumerate(SENTENCES):
        row = [per_ck[j][i] for j in range(len(names))]
        lines.append("| %s | %s | `%s` |" % (
            en, " | ".join("%.1f" % v for v in row), names[row.index(max(row))]))

    lines += ["", "## Verdict", "",
              "| | mean per-sentence BLEU | sentences won |", "|---|---|---|"]
    for j, name in enumerate(names):
        lines.append("| `%s` | %.2f | %d / %d |"
                     % (name, sum(per_ck[j]) / len(per_ck[j]), wins[name], len(SENTENCES)))
    lines += ["", "**%s** wins %d of %d sentences."
              % (max(wins, key=lambda k: wins[k]),
                 wins[max(wins, key=lambda k: wins[k])], len(SENTENCES)), ""]

    out = RESULTS_DIR / ("inference_%s.md" % "_vs_".join(names))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
