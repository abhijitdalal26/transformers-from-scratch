# transformers-from-scratch

Learning transformers and LLMs by building them from scratch in PyTorch.

Each project is a complete, self-contained implementation: the architecture, the
training loop, evaluation, and the visualisations needed to understand what the
model actually learned.

---

## Status

| Project | State | Notes |
|---|---|---|
| **Transformer (en→nl)** | **Done** | 20 epochs on `opus_books`, checkpoint + metrics + graphs in the repo |
| Neural foundations | Done | Perceptron and backpropagation written by hand, no libraries |
| LLaMA | Notes | Reference papers, implementation not started |

### Transformer — English → Dutch

![training dashboard](pytorch-transformer/graphs/training_dashboard.png)

| | |
|---|---|
| Task | English → Dutch machine translation |
| Dataset | `Helsinki-NLP/opus_books` — 38,652 sentence pairs |
| Split | 90 / 10, seeded (`seed=42`) so held-out scores are reproducible |
| Architecture | 6 encoder + 6 decoder layers, 8 heads, `d_model=512`, `d_ff=2048`, dropout 0.1 |
| Tokenizer | WordLevel (`min_frequency=2`) — 17,396 en / 19,637 nl |
| Parameters | **73.50M** (18.96M of them embeddings) |
| Training | 20 epochs, 86,980 steps, batch 8, constant LR 1e-4, label smoothing 0.1 |
| Final train loss | **1.83** (started at 9.93 ≈ `ln(19637)`) |
| **Validation BLEU-4** | **24.21** on 200 held-out sentences (`tmodel_17`) |
| CER / WER | 0.569 / 0.920 |
| Checkpoint | `pytorch-transformer/model/tmodel_17.pt` — 838 MB (weights + Adam moments) |

**The model overfits after epoch 17.** BLEU peaked at **24.21** on epoch 17 and
declined to 24.21 → 20.85 → 20.30 over the last three epochs while training loss
kept falling. More epochs will not help; a regularisation or tokenizer change will.
`tmodel_17` is the checkpoint kept; epochs 18 and 19 are on `drive/` as evidence.

<details>
<summary>Sample translations</summary>

| Source | Reference | Model |
|---|---|---|
| But it was too late. | Doch het was te laat. | Maar het was te laat . |
| "Yes, Ned, he had business with his bankers." | "Ja, Ned, hij moest bij zijn bankier zijn." | " Ja , Ned , hij moest zijn bankier zijn ." |
| CHAPTER 5 Arabian Tunnel | HOOFDSTUK XXIX De Arabische Tunnel. | HOOFDSTUK XVI De Arabische tunnel . |

The model produces grammatical Dutch and gets short sentences right, but BLEU is
capped by the WordLevel tokenizer: with no subwords, `lopende` / `lopend` /
`gelopen` share no representation and unseen words collapse to `[UNK]`.

</details>

---

## Repository layout

```
pytorch-transformer/     the en→nl Transformer (main project)
foundation/              perceptron + backpropagation from scratch
LLaMA/                   LLaMA 2/3 reference papers
imgs/                    figures for foundation/README.md
```

### `pytorch-transformer/` — Transformer

**The implementation** (5 files, 774 lines — this is the model)

| File | What it provides |
|---|---|
| `model.py` | The whole architecture: `MultiHeadAttentionBlock`, `FeedForwardBlock`, `ResidualConnection`, `Encoder`/`Decoder` blocks, sinusoidal `PositionalEncoding`, and `build_transformer()`. Built so each paper equation maps to one class. |
| `dataset.py` | `BilingualDataset` — tokenises, adds `[SOS]/[EOS]/[PAD]`, builds the padding and causal masks, and truncates over-long sentences. Also the `causal_mask()` helper used by decoding. |
| `train.py` | Training loop (Adam, label smoothing, pad-ignored cross-entropy), greedy decode, per-epoch validation, and a self-contained `corpus_bleu()` — `torchmetrics`' BLEU returns 0.0 on this environment, so BLEU-n is computed from scratch with the standard 13a tokenizer. |
| `translate.py` | Single-sentence inference: encodes once, then decodes greedily. Accepts a raw sentence or a dataset index, plus `--checkpoint N`. The 838 MB checkpoint is cached, so translating many sentences in one process loads it once. |
| `config.py` | All hyperparameters in one place, plus checkpoint discovery (`latest_weights_file_path`, `resolve_checkpoint`). Every local path is anchored to this file, not the working directory. |

**`tools/` — everything built around the model** (analysis, scoring, plotting)

These are not part of the model. They import the implementation from the parent
directory and can be run from any working directory.

| File | What it provides |
|---|---|
| `tools/_paths.py` | Path bootstrap: puts the project root on `sys.path` and defines the shared output directories. Every tool imports this first. |
| `tools/evaluate.py` | Scores a checkpoint on the held-out split: BLEU-4 + per-n precisions, CER, WER, exact-match, length stats. Writes `results/wordlevel/metrics_*.json` + `samples_*.md`. |
| `tools/compare_checkpoints.py` | Decodes the same held-out sentences with N checkpoints, writes a side-by-side table plus `graphs/wordlevel/checkpoint_comparison.png`, and picks a per-metric and overall winner. |
| `tools/novel_sentences.py` | 20 sentences written from scratch that appear in neither split — a fairer generalisation read than the validation score. |
| `tools/check_inference.py` | Smoke-tests inference: single word, out-of-vocabulary, punctuation-only, 810-word input. |
| `tools/make_graphs.py` | Rebuilds every training curve from the raw TensorBoard event files. |
| `tools/make_attention_graphs.py` | Renders attention heatmaps (encoder, decoder, cross-attention) from the trained weights. |
| `tools/write_model_card.py` | Regenerates `model/model_card.json` from the checkpoint, config and saved metrics. |

**Superseded notebooks** (kept for reference; the Colab notebook covers all of them)

| File | Superseded by |
|---|---|
| `tools/attention_visual.ipynb` | `colab_transformer_training.ipynb` cells 15-22, and `tools/make_attention_graphs.py` |
| `tools/Beam_Search.ipynb` | `colab_transformer_training.ipynb` cells 22-23 |
| `tools/Inference.ipynb` | `translate.py` |

**Results and artefacts**

| Path | What it provides |
|---|---|
| `colab_transformer_training.ipynb` | The Colab driver — trains on a T4, streams TensorBoard live, then runs inference, attention maps and temperature sampling. Set `SKIP_TRAIN=True` to load the Drive checkpoint instead of retraining. |
| `graphs/` | All training and attention figures (see below). |
| `results/` | Evaluation output: metrics JSON, per-epoch `training_metrics.csv`, and tables of sample translations. |
| `wordlevel_epoch16_samples.md` | Deep-dive on one checkpoint: vocabulary analysis, measured parameter counts, and the tokenizer flaws that cap the score. The baseline to compare a future BPE run against. |
| `results/wordlevel/RUN_REPORT.md` | The full write-up: setup, measured parameter split, the overfitting analysis, generalisation test, and what to change next. |
| `model/` | The trained model: `tmodel_17.pt` (the validated best checkpoint), `tokenizer_en.json` / `tokenizer_nl.json`, and `model_card.json`. The `.pt` and tokenizer JSONs are git-ignored; the model card is tracked. |
| `drive/` | Colab Drive mirror. Holds the TensorBoard event files (the source of truth for every training graph) and the discarded `tmodel_18.pt` / `tmodel_19.pt` checkpoints. |
| `drive/transformers-from-scratch_runs/` | Raw TensorBoard event files. These are the source of truth for every training graph. |

**Reference**

| File | What it provides |
|---|---|
| `pdf/Attention Is All You Need.pdf` | The original Vaswani et al. 2017 paper that this implementation follows. |
| `pdf/Diagrams_V2.pdf` | Umar Jamil's illustrated "Transformer from scratch" notes — the visual companion to this code, in the same order as the classes in `model.py`. |

### `graphs/`

| File | What it shows |
|---|---|
| `training_dashboard.png` | All four training plots on one page — the quickest way to see the overfitting. |
| `train_loss.png` | Per-step training loss on a log scale, with a 500-step mean. Shaded bands are gaps in the log (the run was restarted and some event files were never uploaded). |
| `val_bleu.png` | Validation BLEU-4 per epoch, with the best epoch marked. |
| `val_error_rates.png` | CER and WER per epoch. |
| `bleu_precisions.png` | Per-n precision for BLEU-1…4. |
| `attention/attention_overview.png` | Encoder self-attention, decoder self-attention and cross-attention side by side for one real sentence. |
| `attention/*_layer2_heads.png` | All 8 heads of a single layer, for each of the three attention types. |
| `attention/*_layers_heads.png` | All 6 layers with heads averaged — shows how the pattern sharpens with depth. |

The decoder self-attention plots are strictly lower-triangular, which is the
causal mask working correctly. In the cross-attention plots you can see the
alignment directly (`maatregelen` → `measures`, `nemen` → `take`).

### `foundation/`

| File | What it provides |
|---|---|
| `perceptron-xor-failure-and-solution.ipynb` | A single perceptron built from scratch, shown failing on XOR, then replaced by a 2-layer MLP that solves it. Includes the loss-surface plots. |
| `backpropagation_from_scratch.ipynb` | Forward pass, derivative chain rule and weight updates implemented by hand, then applied to Titanic survival prediction with k-fold cross-validation. |
| `README.md` | The history: perceptron → XOR critique → AI winter → backprop. |

### `LLaMA/`

`LLaMA_Final.pdf`, `LLaMA_2_research_paper.pdf`, `LLaMA_3_research_paper.pdf` —
reference papers. Implementation not started.

---

## Running it

### Locally

```bash
pip install -r pytorch-transformer/requirements.txt

# the model
python pytorch-transformer/translate.py "But it was too late."
python pytorch-transformer/translate.py --checkpoint 17 "The cat sleeps on the table."

# the tools (work from any directory)
python pytorch-transformer/tools/evaluate.py --samples 200          # BLEU / CER / WER
python pytorch-transformer/tools/novel_sentences.py --checkpoint 17  # never-seen sentences
python pytorch-transformer/tools/compare_checkpoints.py \
    --checkpoints pytorch-transformer/model/tmodel_17.pt \
                  pytorch-transformer/drive/tmodel_18.pt \
                  pytorch-transformer/drive/tmodel_19.pt
python pytorch-transformer/tools/check_inference.py                 # edge cases
python pytorch-transformer/tools/make_graphs.py                     # rebuild the curves
python pytorch-transformer/tools/make_attention_graphs.py           # attention heatmaps
python pytorch-transformer/tools/write_model_card.py                # model card

# training
python pytorch-transformer/train.py                 # writes model/tmodel_NN.pt
```

`train.py` also appends one row per epoch to
`results/wordlevel/training_metrics.csv`, so the peak epoch is always
recoverable from a plain text file even if the run dies.
```

### On Colab (recommended)

Open `pytorch-transformer/colab_transformer_training.ipynb` → T4 GPU → Run All.
It mounts Drive, trains with live TensorBoard, then runs inference and attention
visualisation. Set `SKIP_TRAIN=True` to skip the ~3.5h of training and load the
saved checkpoint instead.

---

## Known limitations

Things to fix if this model is picked up again:

- **The tokenizer is the bottleneck.** WordLevel with `min_frequency=2` and no
  subwords: ~44% of the English vocabulary and ~40% of the Dutch vocabulary are
  dead entries, and any unseen word becomes `[UNK]`. Switching to a BPE
  tokenizer is the single highest-value change — `wordlevel_epoch16_samples.md`
  is the baseline to measure it against.
- **No lowercasing.** `zij` and `Zij` are two separate vocabulary entries;
  981 words are stored in both cases.
- **It overfits after epoch 17.** No dropout on the embeddings, no weight
  tying between the target embedding and the projection layer, constant LR with
  no schedule. The NoAM schedule is implemented in `config.py` as `lr_schedule`
  but is opt-in — switching it on mid-run would collapse the LR.
- **Incomplete training log.** The TensorBoard event files are missing steps
  13,186–26,094 and 43,489–73,933 (roughly epochs 3–5 and 10–17) — 46,995 of
  86,980 steps survived, 54%. This was Colab dropping the event file when a
  session disconnected, not a logging bug, so those steps are simply gone and
  the true BLEU peak may be earlier than epoch 17. `tools/make_graphs.py` shades
  the gaps. **Now fixed going forward:** `train.py` also appends one row per
  epoch to `results/wordlevel/training_metrics.csv`, which is a plain file and
  cannot be lost to a disconnect.
- **`config.py` still has `num_epochs: 1`**, left over from a debug run. With
  `preload: "latest"` and `tmodel_17.pt` in `model/`, `train.py` resolves
  `initial_epoch = 18` and `range(18, 1)` is empty, so it is currently a no-op.
  Set `num_epochs` to the epoch you want to train up to before resuming.
- **No KV cache.** Decoding re-runs the whole decoder prefix each step, so
  generation is O(n²). Fine for a 350-token sequence, wasteful in general.
