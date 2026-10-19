"""Render attention heatmaps from a trained checkpoint.

The MultiHeadAttentionBlock stores its own `attention_scores` on every forward
pass, so a single teacher-forced pass through the model gives us all three
attention types for all 6 layers x 8 heads.

Writes into graphs/attention/:
    encoder_layer{L}_heads.png   all 8 heads of one encoder layer
    decoder_layer{L}_heads.png   all 8 heads of one decoder self-attention layer
    cross_layer{L}_heads.png     all 8 heads of one encoder-decoder layer
    encoder_layers_heads.png     every layer, heads averaged
    decoder_layers_heads.png     every layer, heads averaged
    cross_layers_heads.png       every layer, heads averaged
    attention_overview.png       one summary figure

Usage:
    python tools/make_attention_graphs.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _paths import ATTENTION_DIR as OUT  # noqa: F401  (must precede the project imports)
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from train import get_ds

INK = "#1f2933"
MAX_TOKENS = 18  # keep the axes readable
N_LAYERS = 6
N_HEADS = 8


def load_model(config, device):
    ts = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))
    model = build_transformer(
        ts.get_vocab_size(), tt.get_vocab_size(),
        config["seq_len"], config["seq_len"], d_model=config["d_model"],
    ).to(device)
    ckpt = latest_weights_file_path(config)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("loaded", ckpt, "| epoch", state["epoch"])
    return model, ts, tt


def real_lengths(batch):
    """Actual (non-pad) source and target lengths for a batch of 1.

    decoder_mask is (1, seq, seq): the padding mask is broadcast down the rows
    and the causal mask goes across them, so only the LAST row has every real
    position open. Row 0 always has exactly one open position.
    """
    src_len = int(batch["encoder_mask"][0, 0, 0].sum())
    tgt_len = int(batch["decoder_mask"][0, 0, -1].sum())
    return src_len, tgt_len


def pick_sentence(val_dataloader, lo=12, hi=18):
    """Validation sentence with a readable, non-trivial length.

    Scores the first 400 candidates by how close they are to the middle of
    [lo, hi] and returns the best, so a slightly-off length never fails.
    """
    best = None
    best_score = None
    for i, batch in enumerate(val_dataloader):
        if i >= 400:
            break
        src_len, tgt_len = real_lengths(batch)
        if src_len < 4 or tgt_len < 4:
            continue
        mid = (lo + hi) / 2.0
        score = abs(src_len - mid) + abs(tgt_len - mid)
        if best_score is None or score < best_score:
            best, best_score = batch, score
    if best is None:
        best = next(iter(val_dataloader))
    return best, best["src_text"][0], best["tgt_text"][0]


def collect(model, batch, device):
    enc_in = batch["encoder_input"].to(device)
    dec_in = batch["decoder_input"].to(device)
    enc_mask = batch["encoder_mask"].to(device)
    dec_mask = batch["decoder_mask"].to(device)

    with torch.no_grad():
        enc_out = model.encode(enc_in, enc_mask)
        dec_out = model.decode(enc_out, enc_mask, dec_in, dec_mask)
        model.project(dec_out)

    enc = [blk.self_attention_block.attention_scores.detach().cpu().numpy()[0]
           for blk in model.encoder.layers]
    dec = [blk.self_attention_block.attention_scores.detach().cpu().numpy()[0]
           for blk in model.decoder.layers]
    cross = [blk.cross_attention_block.attention_scores.detach().cpu().numpy()[0]
             for blk in model.decoder.layers]
    return enc, dec, cross


def draw_grid(maps, row_toks, col_toks, path, title, cmap="viridis"):
    """maps: (n_panels, heads, q, k)"""
    n_panels, heads = maps.shape[0], maps.shape[1]
    fig, axes = plt.subplots(n_panels, heads,
                             figsize=(heads * 2.3, n_panels * 2.5), squeeze=False)
    for r in range(n_panels):
        for h in range(heads):
            ax = axes[r][h]
            m = maps[r, h]
            im = ax.imshow(m, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(col_toks)))
            ax.set_xticklabels(col_toks, rotation=90, fontsize=6)
            ax.set_yticks(range(len(row_toks)))
            ax.set_yticklabels(row_toks, fontsize=6)
            if r == 0:
                ax.set_title("head %d" % h, fontsize=9, color=INK)
            if h == 0:
                ax.set_ylabel("layer %d" % r, fontsize=9, color=INK)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12, color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def draw_layers(maps, row_toks, col_toks, path, title, cmap="viridis"):
    """maps: (layers, heads, q, k) -> one panel per layer, heads averaged."""
    layers = maps.shape[0]
    fig, axes = plt.subplots(1, layers, figsize=(layers * 3.0, 3.6), squeeze=False)
    for l in range(layers):
        ax = axes[0][l]
        m = maps[l].mean(axis=0)
        im = ax.imshow(m, cmap=cmap, aspect="auto")
        ax.set_title("layer %d" % l, fontsize=10, color=INK)
        ax.set_xticks(range(len(col_toks)))
        ax.set_xticklabels(col_toks, rotation=90, fontsize=6)
        ax.set_yticks(range(len(row_toks)))
        ax.set_yticklabels(row_toks, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=12, color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = get_config()
    device = torch.device("cpu")

    model, ts, tt = load_model(config, device)
    _, val_dataloader, _, _ = get_ds(config)
    batch, src, tgt = pick_sentence(val_dataloader)
    print("SOURCE:", src)
    print("TARGET:", tgt)

    enc, dec, cross = collect(model, batch, device)

    # trim to the real (non-pad) length so the axes are not mostly [PAD]
    src_len, tgt_len = real_lengths(batch)
    n_src = min(src_len, MAX_TOKENS)
    n_tgt = min(tgt_len, MAX_TOKENS)

    src_ids = batch["encoder_input"][0].tolist()[:n_src]
    tgt_ids = batch["decoder_input"][0].tolist()[:n_tgt]
    src_toks = [ts.id_to_token(i) for i in src_ids]
    tgt_toks = [tt.id_to_token(i) for i in tgt_ids]
    src_toks = [t if t else "_" for t in src_toks]
    tgt_toks = [t if t else "_" for t in tgt_toks]

    # (layers, heads, q, k) trimmed per attention type
    enc_maps = np.stack([e[:, :n_src, :n_src] for e in enc])
    dec_maps = np.stack([d[:, :n_tgt, :n_tgt] for d in dec])
    cross_maps = np.stack([c[:, :n_tgt, :n_src] for c in cross])

    LAYER = 2
    draw_grid(enc_maps[LAYER:LAYER + 1], src_toks, src_toks,
              OUT / f"encoder_layer{LAYER}_heads.png",
              f"Encoder self-attention - layer {LAYER}, all 8 heads (rows = source tokens)")
    draw_grid(dec_maps[LAYER:LAYER + 1], tgt_toks, tgt_toks,
              OUT / f"decoder_layer{LAYER}_heads.png",
              f"Decoder self-attention - layer {LAYER}, all 8 heads (causal)")
    draw_grid(cross_maps[LAYER:LAYER + 1], tgt_toks, src_toks,
              OUT / f"cross_layer{LAYER}_heads.png",
              f"Encoder-decoder (cross) attention - layer {LAYER}, all 8 heads")

    draw_layers(enc_maps, src_toks, src_toks, OUT / "encoder_layers_heads.png",
                "Encoder self-attention - 6 layers, heads averaged")
    draw_layers(dec_maps, tgt_toks, tgt_toks, OUT / "decoder_layers_heads.png",
                "Decoder self-attention - 6 layers, heads averaged (causal)")
    draw_layers(cross_maps, tgt_toks, src_toks, OUT / "cross_layers_heads.png",
                "Encoder-decoder attention - 6 layers, heads averaged")

    # summary: diagonal of the three types at one layer
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, m, t, name in (
        (axes[0], enc_maps[LAYER].mean(0), src_toks, "encoder self-attn"),
        (axes[1], dec_maps[LAYER].mean(0), tgt_toks, "decoder self-attn (causal)"),
        (axes[2], cross_maps[LAYER].mean(0), src_toks, "encoder-decoder (cross)"),
    ):
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set_title("%s - layer %d" % (name, LAYER), fontsize=10, color=INK)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels(t, rotation=90, fontsize=7)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels(t, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Attention patterns - tmodel_19 (heads averaged)\nSOURCE: %s" % src[:110],
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "attention_overview.png", dpi=140)
    plt.close(fig)
    print("wrote", OUT / "attention_overview.png")


if __name__ == "__main__":
    main()
