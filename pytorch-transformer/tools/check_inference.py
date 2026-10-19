"""Smoke-test inference: normal sentences, edge cases and known failure modes.

Run after a checkpoint is in place:
    python check_inference.py
"""

import contextlib
import io
import traceback

import torch
from tokenizers import Tokenizer

from _paths import MODEL_DIR  # noqa: F401  (must precede the project imports)
from config import get_config, latest_weights_file_path
from model import build_transformer
from translate import translate

CASES = [
    ("normal", "I am not a very good a student."),
    ("short", "Hello."),
    ("one word", "Water."),
    ("question", "Why do I need to translate this?"),
    ("long-ish", "It was the best of times, it was the worst of times, it was the age of "
                 "wisdom, it was the age of foolishness, and nobody could tell the difference."),
    ("oov / rare", "Zzyzx qwertzuiop blorptastic."),
    ("punctuation only", "... !!! ???"),
    ("numbers", "In 1789 the fleet had 12 ships and 4500 sailors."),
    ("very long", " ".join(["the quick brown fox jumps over the lazy dog"] * 90)),
]


def main():
    config = get_config()
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("checkpoint:", latest_weights_file_path(config))
    print("-" * 70)

    failures = []
    for name, text in CASES:
        n_words = len(text.split())
        try:
            # translate() prints its own progress; capture it so this report stays readable
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = translate(text, config)
            status = "OK"
            preview = (out or "").strip()
            if not preview:
                status = "EMPTY"
            print("[%-7s] %-13s (%3d words) -> %s" % (status, name, n_words, preview[:95]))
            if status != "OK":
                failures.append((name, status))
        except Exception as e:
            msg = "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:110])
            print("[FAIL   ] %-13s (%3d words) -> %s" % (name, n_words, msg))
            failures.append((name, msg))
            if "-v" in __import__("sys").argv:
                traceback.print_exc()

    print("-" * 70)
    print("tokenizer round-trip / vocab check")
    ts = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))
    print("  src vocab:", ts.get_vocab_size(), " tgt vocab:", tt.get_vocab_size())
    for tok in ("[UNK]", "[PAD]", "[SOS]", "[EOS]"):
        print("  tgt %-6s id=%s" % (tok, tt.token_to_id(tok)))
    unk = tt.token_to_id("[UNK]")
    oov = tt.encode("zzqqxx blorptastic").ids
    print("  oov sentence ids:", oov, "-> all [UNK]?", set(oov) == {unk})

    print("-" * 70)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for n, m in failures:
            print("  -", n, "->", m)
    else:
        print("all inference cases passed")


if __name__ == "__main__":
    main()
