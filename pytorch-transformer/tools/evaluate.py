"""Offline evaluation for a trained checkpoint.

Runs greedy decoding over the *seeded* held-out split (same seed as training,
so the numbers are comparable) and reports BLEU-4, CER and WER, then writes
both the metrics and a set of sample translations to disk.

Usage:
    python evaluate.py                       # eval latest checkpoint, 200 sentences
    python evaluate.py --samples 50          # faster, noisier
    python evaluate.py --checkpoint tmodel_19.pt
    python evaluate.py --device cuda

Outputs (all under results/):
    metrics_<checkpoint>.json   BLEU/CER/WER + per-n precisions + BP
    samples_<checkpoint>.md     SOURCE / TARGET / PREDICTED table
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from _paths import RESULTS_DIR  # noqa: F401  (must precede the project imports)
from config import get_config, latest_weights_file_path, get_weights_file_path, _model_dir, resolve_checkpoint
from model import build_transformer
from train import corpus_bleu, get_ds, greedy_decode


def build_model_and_tokenizers(config, checkpoint, device):
    tokenizer_src = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tokenizer_tgt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, tokenizer_src, tokenizer_tgt, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="e.g. tmodel_19.pt (default: latest)")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--num-print", type=int, default=25, help="how many samples to write")
    args = p.parse_args()

    config = get_config()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print("Using device:", device)

    checkpoint = (
        resolve_checkpoint(config, args.checkpoint)
        if args.checkpoint
        else latest_weights_file_path(config)
    )
    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found. Train the model first.")
    print("Checkpoint:", checkpoint)

    model, tokenizer_src, tokenizer_tgt, state = build_model_and_tokenizers(config, checkpoint, device)
    print("checkpoint epoch =", state["epoch"], " global_step =", state["global_step"])
    print("vocab: src =", tokenizer_src.get_vocab_size(), " tgt =", tokenizer_tgt.get_vocab_size())

    # Rebuild the exact same seeded train/val split used during training.
    _, val_dataloader, _, _ = get_ds(config)

    n = min(args.samples, len(val_dataloader.dataset))
    print(f"decoding {n} held-out sentences (val split size = {len(val_dataloader.dataset)})")

    src_texts, expected, predicted, gen_lens = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="decoding", total=n)):
            if i >= n:
                break
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            out = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, config["seq_len"], device,
            )
            gen_lens.append(len(out))
            src_texts.append(batch["src_text"][0])
            expected.append(batch["tgt_text"][0])
            predicted.append(tokenizer_tgt.decode(out.detach().cpu().numpy()))

    elapsed = time.time() - t0

    # --- metrics -------------------------------------------------------------
    bleu, precisions, bp = corpus_bleu(predicted, expected)

    import torchmetrics
    cer = float(torchmetrics.CharErrorRate()(predicted, expected))
    wer = float(torchmetrics.WordErrorRate()(predicted, expected))
    exact = sum(1 for p, e in zip(predicted, expected) if p.strip() == e.strip())

    stem_rates = []
    tok_s, tok_e = tokenizer_src, tokenizer_tgt
    for p_, e_ in zip(predicted, expected):
        ps = tok_e.encode(p_).ids
        es = tok_e.encode(e_).ids
        if not es:
            continue
        common = 0
        pool = list(es)
        for t in ps:
            if t in pool:
                pool.remove(t)
                common += 1
        stem_rates.append(common / len(es))

    metrics = {
        "checkpoint": Path(checkpoint).name,
        "checkpoint_epoch": state["epoch"],
        "global_step": state["global_step"],
        "device": str(device),
        "n_sentences": len(predicted),
        "val_split_size": len(val_dataloader.dataset),
        "seed": config.get("seed"),
        "bleu4": round(bleu, 4),
        "bleu_precisions": [round(x, 4) for x in precisions],
        "brevity_penalty": round(bp, 4),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact_match": exact,
        "exact_match_pct": round(100.0 * exact / max(len(predicted), 1), 2),
        "mean_unigram_recall": round(statistics.fmean(stem_rates) if stem_rates else 0.0, 4),
        "mean_gen_len_tokens": round(statistics.fmean(gen_lens) if gen_lens else 0.0, 2),
        "empty_predictions": sum(1 for p_ in predicted if not p_.strip()),
        "decode_seconds_total": round(elapsed, 1),
        "decode_seconds_per_sentence": round(elapsed / max(len(predicted), 1), 2),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # include the sample count so a quick smoke run cannot overwrite a full one
    stem = "%s_n%d" % (Path(checkpoint).stem, len(predicted))
    (RESULTS_DIR / f"metrics_{stem}.json").write_text(json.dumps(metrics, indent=2))

    # --- samples -------------------------------------------------------------
    order = sorted(range(len(predicted)), key=lambda i: abs(len(predicted[i]) - len(expected[i])))
    show = order[: args.num_print]
    lines = [
        f"# Greedy translations - {stem}",
        "",
        f"- **Checkpoint:** `{Path(checkpoint).stem}` (epoch {state['epoch']}, global_step {state['global_step']})",
        f"- **Decoding:** greedy, no beam search | **device:** {device}",
        f"- **Scored on:** {len(predicted)} held-out sentences (seed {config.get('seed')})",
        f"- **BLEU-4:** {bleu:.2f} | **CER:** {cer:.3f} | **WER:** {wer:.3f} | **BP:** {bp:.3f}",
        "",
        "Samples are sorted by closest length to the reference (easiest cases first).",
        "",
    ]
    for k, i in enumerate(show, start=1):
        lines += [
            f"### {k}",
            "",
            f"- **SOURCE:** {src_texts[i]}",
            f"- **TARGET:** {expected[i]}",
            f"- **PREDICTED:** {predicted[i] or '(empty)'}",
            "",
        ]
    (RESULTS_DIR / f"samples_{stem}.md").write_text("\n".join(lines))
    print("\n" + "=" * 62)
    for k, v in metrics.items():
        print(f"{k:28s} {v}")
    print("=" * 62)
    print(f"wrote {RESULTS_DIR / ('metrics_' + stem + '.json')}")
    print(f"wrote {RESULTS_DIR / ('samples_' + stem + '.json'.replace('.json', '.md'))}")


if __name__ == "__main__":
    main()
