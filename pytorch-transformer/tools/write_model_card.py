"""Write a model card for the current checkpoint.

Reads the checkpoint, config and the saved evaluation metrics, and writes
`model/model_card.json` so the model directory is self-describing.

Usage:
    python tools/write_model_card.py
"""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from _paths import RESULTS_DIR  # noqa: F401  (must precede the project imports)
from config import get_config, latest_weights_file_path, _model_dir


def main():
    config = get_config()
    model_dir = Path(_model_dir(config))
    ckpt = latest_weights_file_path(config)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    ts = Tokenizer.from_file(str(model_dir / "tokenizer_en.json"))
    tt = Tokenizer.from_file(str(model_dir / "tokenizer_nl.json"))

    params = sum(v.numel() for v in state["model_state_dict"].values())
    # keys are src_embed.embedding.weight / tgt_embed.embedding.weight
    emb = sum(
        v.numel()
        for k, v in state["model_state_dict"].items()
        if k.endswith("embed.embedding.weight")
    )

    metrics_file = RESULTS_DIR / ("metrics_%s_n200.json" % Path(ckpt).stem)
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None

    card = {
        "name": "tmodel",
        "task": "English -> Dutch machine translation",
        "description": (
            "Transformer encoder-decoder trained from scratch on Helsinki-NLP/opus_books "
            "with a WordLevel tokenizer. This is the validated best checkpoint; the run "
            "overfits from epoch 18 onward."
        ),
        "checkpoint": {
            "file": Path(ckpt).name,
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "size_bytes": Path(ckpt).stat().st_size,
            "contains": ["model_state_dict", "optimizer_state_dict", "epoch", "global_step"],
        },
        "architecture": {
            "d_model": config["d_model"],
            "encoder_layers": 6,
            "decoder_layers": 6,
            "attention_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,
            "seq_len": config["seq_len"],
            "parameters": params,
            "embedding_parameters": emb,
            "embedding_share_pct": round(100.0 * emb / params, 2),
        },
        "tokenizer": {
            "type": "WordLevel",
            "pre_tokenizer": "Whitespace",
            "normalizer": None,
            "min_frequency": 2,
            "special_tokens": ["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            "vocab_size_en": ts.get_vocab_size(),
            "vocab_size_nl": tt.get_vocab_size(),
        },
        "training": {
            "dataset": config["datasource"],
            "pairs": 38652,
            "split": "90/10",
            "seed": config["seed"],
            "batch_size": config["batch_size"],
            "steps_per_epoch": 4349,
            "epochs_completed": state["epoch"] + 1,
            "lr": config["lr"],
            "lr_schedule": config["lr_schedule"],
            "label_smoothing": 0.1,
            "optimizer": "Adam (eps=1e-9)",
            "final_train_loss": 1.8316,
            "initial_train_loss": 9.9274,
            "device_trained_on": "Colab T4 GPU",
        },
        "evaluation": {
            "protocol": "greedy decoding, no beam search",
            "scored_on": "held-out validation split, seeded split, first %s sentences"
            % (metrics["n_sentences"] if metrics else "N"),
            "metrics": metrics,
        },
        "overfitting": {
            "onset_epoch": 18,
            "best_epoch": 17,
            "best_bleu4": 24.2076,
            "final_bleu4": 20.3011,
            "note": (
                "BLEU-4 peaked at epoch 17 and fell for the two remaining epochs while "
                "training loss kept dropping. The epoch-19 checkpoint was discarded. "
                "See results/wordlevel/RUN_REPORT.md."
            ),
        },
        "usage": {
            "translate": 'python translate.py "But it was too late."',
            "translate_epoch17": 'python translate.py --checkpoint 17 "The cat sleeps on the table."',
            "evaluate": "python tools/evaluate.py --samples 200",
            "compare": "python tools/compare_checkpoints.py --samples 60",
            "novel_sentences": "python tools/novel_sentences.py --checkpoint 17",
            "note": (
                "preload is 'latest', so with a single checkpoint in model/ all "
                "scripts resolve to it automatically. The tools/ scripts import this "
                "project from the parent directory and can be run from any working "
                "directory. The Colab notebook overrides model_folder and "
                "tokenizer_file to point at Google Drive."
            ),
        },
        "known_limitations": [
            "WordLevel tokenizer with no subwords: unseen words collapse to [UNK], and inflected forms share no representation.",
            "No lowercasing, so 'zij' and 'Zij' are separate vocabulary entries.",
            "~44% of the English and ~40% of the Dutch vocabulary entries are never used in the corpus.",
            "No weight tying between the target embedding and the projection layer.",
            "Constant learning rate, no schedule; the NoAM schedule exists but is opt-in.",
            "No KV cache: decoding is O(n^2) in the generated length.",
        ],
    }

    out = model_dir / "model_card.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    print("wrote", out)
    print("  checkpoint :", card["checkpoint"]["file"], "epoch", card["checkpoint"]["epoch"])
    print("  parameters : %.2fM" % (params / 1e6))
    print("  BLEU-4     :", card["evaluation"]["metrics"]["bleu4"] if metrics else "n/a")


if __name__ == "__main__":
    main()
