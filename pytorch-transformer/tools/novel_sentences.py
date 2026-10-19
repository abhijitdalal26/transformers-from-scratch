"""Translate sentences that appear nowhere in the training corpus.

The validation split is held out, but it is still the same 19th-century book
corpus the model was trained on. These sentences are written from scratch and
appear in neither the train nor the validation split, so they are a fairer
read on real generalisation.

Run with a particular checkpoint:
    python novel_sentences.py
    python novel_sentences.py --checkpoint tmodel_17.pt

Writes results/wordlevel/novel_sentences.md and prints the table.
"""

import argparse
import contextlib
import io
from pathlib import Path

from _paths import RESULTS_DIR  # noqa: F401  (must precede the project imports)
from config import get_config, resolve_checkpoint
from translate import translate

# (english, my own Dutch reference, what it probes)
SENTENCES = [
    ("The cat sleeps on the table.", "De kat slaapt op de tafel.", "simple subject-verb-object"),
    ("I drink coffee every morning.", "Ik drink elke morgen koffie.", "frequency + habitual"),
    ("She does not like fish.", "Zij houdt niet van vis.", "negation"),
    ("We were very happy yesterday.", "Wij waren gisteren zeer gelukkig.", "past tense + intensifier"),
    ("The book is on the shelf.", "Het boek ligt op de plank.", "locative"),
    ("He is taller than his brother.", "Hij is langer dan zijn broer.", "comparative"),
    ("What are you doing?", "Wat doe je?", "question"),
    ("Where is the station?", "Waar is het station?", "question + place"),
    ("I have two brothers and one sister.", "Ik heb twee broers en een zuster.", "numbers + family"),
    ("The weather is cold today.", "Het weer is koud vandaag.", "weather expression"),
    ("They will come to the party.", "Zij zullen naar het feest komen.", "future tense"),
    ("This is a very interesting book.", "Dit is een zeer interessant boek.", "attributive adjective"),
    ("The child is playing in the garden.", "Het kind speelt in de tuin.", "present continuous"),
    ("I do not understand this word.", "Ik begrijp dit woord niet.", "negation + separable verb"),
    ("The man who came yesterday is my friend.",
     "De man die gisteren kwam is mijn vriend.", "relative clause"),
    ("Please open the window.", "Open alsjeblieft het raam.", "polite imperative"),
    ("The colour of the sky is blue.", "De kleur van de hemel is blauw.", "nested possessives"),
    ("Water boils at one hundred degrees.", "Water kookt bij honderd graden.", "numbers as words"),
    ("She gave me a beautiful gift.", "Zij gaf mij een prachtig geschenk.", "dative + adjective"),
    ("I have never seen such a large dog.", "Ik heb nog nooit zo'n grote hond gezien.", "negation + superlative"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    config = get_config()
    ckpt = resolve_checkpoint(config, args.checkpoint) if args.checkpoint else None
    label = Path(ckpt).stem if ckpt else "latest"

    rows = []
    for en, nl, probe in SENTENCES:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = translate(en, config, checkpoint=ckpt)
        pred = (out or "").strip()
        rows.append((en, nl, pred, probe))

    w_en = max(len(r[0]) for r in rows)
    w_nl = max(len(r[1]) for r in rows)
    w_pr = max(len(r[2]) for r in rows)

    print("=" * (w_en + w_nl + w_pr + 24))
    print("NEVER-SEEN SENTENCES  (checkpoint: %s)" % label)
    print("=" * (w_en + w_nl + w_pr + 24))
    for en, nl, pred, probe in rows:
        print("EN  %s" % en.ljust(w_en))
        print("REF %s" % nl.ljust(w_nl))
        print("OUT %s" % pred.ljust(w_pr))
        print("    [%s]" % probe)
        print("-" * (w_en + w_nl + w_pr + 24))

    empties = sum(1 for r in rows if not r[2])
    # crude lexical overlap with the reference as a sanity signal only
    hits = 0
    for _, nl, pred, _ in rows:
        pset = {w.lower().strip(".,!?;:") for w in pred.split() if w.strip(".,!?;:")}
        rset = {w.lower().strip(".,!?;:") for w in nl.split() if w.strip(".,!?;:")}
        if pset and rset and len(pset & rset) / len(rset) >= 0.5:
            hits += 1
    print()
    print("sentences        : %d" % len(rows))
    print("empty output     : %d" % empties)
    print(">=50%% ref words  : %d / %d" % (hits, len(rows)))

    out = RESULTS_DIR / "novel_sentences.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Never-seen sentences - `%s`" % label,
        "",
        "Sentences written from scratch for this test. They appear in neither the",
        "training split nor the validation split of `Helsinki-NLP/opus_books`, so this is",
        "a fairer read on generalisation than the validation score. Dutch references are",
        "my own translations, not corpus gold.",
        "",
        "| English | Dutch reference | Model output | Probes |",
        "|---|---|---|---|",
    ]
    for en, nl, pred, probe in rows:
        lines.append("| %s | %s | %s | %s |" % (en, nl, pred or "*(empty)*", probe))
    lines += [
        "",
        "## Summary",
        "",
        "- sentences: %d" % len(rows),
        "- empty outputs: %d" % empties,
        "- outputs sharing >=50%% of the reference words: %d / %d" % (hits, len(rows)),
        "",
    ]
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
