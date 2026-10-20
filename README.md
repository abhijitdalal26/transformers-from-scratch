# Transformer from scratch — English → Dutch

One project. A full encoder-decoder Transformer, written from scratch in PyTorch,
trained to translate English into Dutch.

The point of the project was to **learn the transformer architecture** by
implementing every part of it and watching what it actually does when you
train it. The architecture is implemented in
[`pytorch-transformer/model.py`](pytorch-transformer/model.py) in roughly 227
lines, and the whole training pipeline around it in about 1,060 lines of
Python. The dataset is small (38,652 sentence pairs), the model is small
(73.1M parameters), and the hardware was a single Colab T4 — so this is not a
state-of-the-art translation system. It is a complete, honest, instrumented
implementation of the paper, plus a record of what the experiment actually
taught.

Everything else in this repository is study material: the fundamentals that
came before the transformer, and the architectures I want to build next.

---

## The result

| | |
|---|---|
| **Model** | `pytorch-transformer/model/tmodel_17.pt` (epoch 17, global step 78,282) |
| **Architecture** | 6 encoder + 6 decoder layers, d_model 512, 8 heads, d_ff 2048, dropout 0.1, sinusoidal positional encoding |
| **Parameters** | 73,138,357 |
| **Tokenizer** | WordLevel, 17,396 English / 19,637 Dutch words, `[UNK] [PAD] [SOS] [EOS]` |
| **Data** | `Helsinki-NLP/opus_books`, 38,652 pairs → 34,786 train / 3,866 val (seeded 90/10 split, seed 42) |
| **Training** | 20 epochs, 4,349 steps/epoch, 86,980 steps, batch 8, Adam (eps 1e-9), constant LR 1e-4, label smoothing 0.1, Colab T4 |
| **Scored on** | the first 200 held-out validation sentences, greedy decoding |

| Metric | Value |
|---|---|
| BLEU-4 (cased) | **24.2076** |
| BLEU-4 (case-blind) | **24.4271** |
| Precision p1 / p2 / p3 / p4 | 0.5372 / 0.2850 / 0.1824 / 0.1294 |
| Brevity penalty | 0.9873 |
| CER / WER | 0.5686 / 0.9196 |
| Empty predictions | 0 / 200 |

What it produces, on sentences it has never seen:

| English | Dutch (model) |
|---|---|
| Where is the station? | Waar is het station ? |
| We were very happy yesterday. | Wij waren gisteren al zeer gelukkig . |
| I have two brothers and one sister. | Ik heb twee broeders en een zuster . |
| But it was too late. | Maar het was te laat . |
| Joe took a pin out of his lapel … | Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje te geven . |

26 more (20 never-seen sentences plus 6 held-out ones, across four
checkpoints) are in
[`results/comparison/bpe_sweep/inference_showcase.md`](pytorch-transformer/results/comparison/bpe_sweep/inference_showcase.md).

---

## Results and artefacts

The full write-up is
[`pytorch-transformer/results/FINAL_REPORT.md`](pytorch-transformer/results/FINAL_REPORT.md).
Everything below is a summary of what the run showed.

### The overfitting curve

Validation BLEU-4 rose for 17 epochs and then fell off a cliff, while the
training loss kept going down. That gap between "it fits the training data
better and better" and "it is getting worse at the real task" is the single
most useful thing this project produced.

| Epoch | BLEU-4 | CER | WER | train loss |
|---|---|---|---|---|
| **17** | **24.21** | 0.569 | 0.920 | ~2.5, falling |
| 18 | 20.85 | 0.580 | 0.921 | falling |
| 19 | 20.30 | 0.591 | 0.937 | 1.83 |

BLEU fell 3.91 points (16% relative) over the last two epochs. The precisions
show *how* it degraded: p1 only dropped 2.1 points (0.5372 → 0.5160) while p4
dropped 3.1 points, 24% relative (0.1294 → 0.0988). The model kept choosing
the right *words* and lost the *ordering*. Generated length stayed flat
(24.77 → 24.52), so it was not learning to give up early either.

The lesson: the last two epochs of compute were wasted, and a 34,786-pair
dataset with a 73M-parameter model is already past the point where more
training helps. Stopping at the peak matters.

### The BPE run, and what it taught

I re-ran the whole experiment with a BPE tokenizer (8,000 merge-based
subwords, 56.40M parameters) to test the hypothesis that the WordLevel
tokenizer was the ceiling on quality. **That hypothesis was wrong, and the
experiment is more interesting than the original project.**

| | WordLevel ep17 | BPE ep9 (its own best) | BPE ep17 | BPE ep19 |
|---|---|---|---|---|
| BLEU-4 cased | **24.2076** | 8.5951 | 8.0134 | 7.7222 |
| BLEU-4 case-blind | **24.4271** | 11.5511 | 10.9415 | 10.1922 |
| generalisation gap (nats/token) | **0.474** | 1.784 | 3.779 | 4.028 |
| source dependence | **81.2%** | 43.8% | 42.0% | 38.5% |

Two independent measurements show the BPE model was **memorising**, not
translating:

1. **Generalisation gap.** Teacher-forced loss on the training set versus on
   unseen data. WordLevel's gap is 0.474 nats/token. BPE's gap is already
   1.784 nats/token at its *best* checkpoint — 3.8× worse — and grows to
   4.028 (8.5×) by epoch 19 while its training loss keeps falling.
2. **Source dependence.** Feed the encoder a shuffled or blanked-out source
   and see how much of the score survives. WordLevel loses 81.2% of its
   performance, so it is genuinely reading the English sentence. BPE starts at
   43.8% and *falls monotonically* to 38.5% as it trains. The model was
   progressively **stopping reading its input** and emitting a generic
   average Dutch sentence instead.

The tokenizer itself did its job perfectly: unknown-word rate went from
1.27% to 0.00% on the source and 1.60% to 0.00% on the reference, and the
vocabulary shrank from 37,033 entries to 16,000. And the comparison was
actually *generous* to BPE, because its tokenizers were built from the whole
corpus including the validation split (`train.py:333-334`) while the
WordLevel tokenizer was built from the training split only.

The mechanism is plain: BPE splits a word into ~1.3 pieces, so it gets
232,016 target tokens per epoch instead of WordLevel's 205,452 — 26,564 extra
free supervision targets per epoch on a corpus of only 34,786 pairs. More
targets to memorise, and a vocabulary that can represent any of them.

**What I take from it:** BPE is not worse, it is *underfed*. It won on every
axis that matters for translation (no unknowns, shorter sequences, subword
morphology) and still lost, purely because there was not enough data to
learn it. The moment this same BPE model is trained on a million-plus pairs
instead of 35 thousand, with a learning-rate schedule and early stopping, it
is the one that should win — the subword vocabulary generalises to unseen
words, and the WordLevel vocabulary structurally cannot. **The tokenizer was
never the bottleneck. The dataset size was.**

### The graphs

Generated by `tools/make_graphs.py` and `tools/make_comparison_graphs.py`.
They are local-only (see [Repository layout](#repository-layout)) and can be
regenerated at any time.

| Graph | What it shows | What it told me |
|---|---|---|
| `graphs/wordlevel/train_loss.png` | per-step training loss, log scale, with the run restarts shaded | The loss never plateaued — it just kept descending. Training loss is a bad stopping signal on its own. |
| `graphs/wordlevel/val_bleu.png` | validation BLEU-4 per epoch, best epoch marked | The peak at epoch 17, and the drop after it. The only curve that matters. |
| `graphs/wordlevel/val_error_rates.png` | CER and WER per epoch | WER > 1.0 means it produced more wrong words than the reference had words; the model was often structurally wrong, not just lexically. |
| `graphs/wordlevel/bleu_precisions.png` | BLEU n-gram precisions p1–p4 | Short n-grams degrade gently, long n-grams collapse. Overfitting shows up first in word *order*. |
| `graphs/wordlevel/overfitting.png` | training loss and validation BLEU on one axis | The divergence point, visually. |
| `graphs/wordlevel/checkpoint_comparison.png` | epochs 17 / 18 / 19 side by side | Epoch 17 tracks clause structure faithfully; epoch 19 degenerates. |
| `graphs/wordlevel/training_dashboard.png` | the four panels above combined | The one-glance summary of the whole run. |
| `graphs/wordlevel/attention/*.png` | attention heatmaps: encoder self-attention, decoder self-attention, cross-attention | The mechanism made visible. Self-attention shows what each head latches onto inside a language; cross-attention shows how the decoder lines up an English word with its Dutch counterpart. Reading these is the fastest way to see that the model learned structure rather than a lookup table. |
| `graphs/bpe/*.png` | the same panel set for the BPE run | The same shape, but the validation curve peaks at epoch 9 and never comes back. |
| `graphs/comparison/val_bleu_vs_epochs.png` | both runs' validation BLEU per epoch | The gap opens immediately and never closes. |
| `graphs/comparison/overfitting_vs_epochs.png` | "Where each run starts overfitting" | WordLevel turns at epoch 18; BPE was already turning at its own peak, epoch 9. BPE memorised 8 epochs earlier. |
| `graphs/comparison/memorisation_evidence.png` | teacher-forced loss per token on train vs val, per checkpoint, with the gap annotated | The bar chart version of the memorisation proof above. |

### Everything else in `results/`

| File | What is in it |
|---|---|
| `FINAL_REPORT.md` | the main write-up: scoreboard, which checkpoint is best, why BPE lost, the architecture walkthrough, what to do next |
| `wordlevel/RUN_REPORT.md` | the WordLevel run in detail, epoch by epoch |
| `wordlevel/model_card.json` | the model card: architecture, tokenizer, training config, evaluation results, known limitations |
| `comparison/BPE_VS_WORDLEVEL.md` | the tokenizer comparison, with the correction that the best BPE checkpoint is epoch 9 |
| `comparison/bpe_sweep/SWEEP.md` | all 8 checkpoints scored with one protocol |
| `comparison/bpe_sweep/overfit_sweep.md` | the generalisation-gap table |
| `comparison/bpe_sweep/source_ablation.md` | the source-dependence table |
| `comparison/bpe_sweep/inference_showcase.md` | 26 sentences × 4 checkpoints, side by side |
| `bpe/training_metrics.csv` | the BPE run's per-epoch metrics, and `training_metrics.NOTES.md` explaining exactly which rows are original and which were reconstructed |

---

## The implementation

Five files, about 1,060 lines. This is the whole project.

| File | Lines | What it holds |
|---|---|---|
| [`model.py`](pytorch-transformer/model.py) | 227 | **The transformer.** `PositionalEncoding` (sinusoidal), `InputEmbeddings`, `MultiHeadAttentionBlock`, `FeedForwardBlock`, `ResidualConnection`, `LayerNormalization`, `EncoderBlock` / `DecoderBlock`, `Encoder`, `Decoder`, `ProjectionLayer`, and the `Transformer` class that wires them together. Causal masking for the decoder, padding masks for the encoder, everything from the paper, no library. |
| [`dataset.py`](pytorch-transformer/dataset.py) | 80 | `BilingualDataset` — tokenises the parallel corpus, adds `[SOS]`/`[EOS]`/`[PAD]`, builds the causal and padding masks, truncates over-long sentences, and yields batches. Plus `causal_mask(size)`. |
| [`config.py`](pytorch-transformer/config.py) | 152 | Every hyperparameter in one place (d_model, layers, heads, d_ff, dropout, seq_len, LR, batch size, label smoothing, epochs) and the checkpoint-resolution logic: where weights live, which tokenizer to load, which epoch to use. Every other file anchors its paths here, so the model folder can be moved without touching anything else. |
| [`train.py`](pytorch-transformer/train.py) | 505 | The training loop, the validation pass, greedy decoding, and the metrics. Includes a hand-written `corpus_bleu` (torchmetrics' BLEU returns 0.0 in this environment) and a per-epoch metrics CSV. |
| [`translate.py`](pytorch-transformer/translate.py) | 97 | Single-sentence inference. Encodes once, then decodes greedily, caching the 838 MB checkpoint so repeated calls are fast. Takes a raw sentence or a dataset index, and `--checkpoint N` to pick an epoch. |

---

## Repository layout

```
neural-network-projects/
├── pytorch-transformer/        ← the project
│   ├── model.py               the transformer
│   ├── dataset.py             tokenisation + batching
│   ├── config.py              hyperparameters + checkpoint resolution
│   ├── train.py               training loop + validation
│   ├── translate.py           inference
│   ├── requirements.txt
│   ├── colab_transformer_training.ipynb
│   ├── results/               the tracked record: reports, metrics, samples
│   ├── pdf/                   "Attention Is All You Need" + the paper diagrams
│   ├── model/                 tmodel_17.pt + tokenizers      (local only)
│   ├── drive_bpe/             BPE weights + per-epoch CSV   (local only)
│   ├── graphs/                generated plots                (local only)
│   ├── tools/                 analysis and plotting scripts  (local only)
│   └── runs/                  TensorBoard event files        (local only)
├── karpathy/                  reference material: BPE, makemore, LLaMA
├── foundation/                perceptron + backprop notebooks
├── LLaMA/                     LLaMA 1/2/3 papers
└── imgs/                      figures for foundation/
```

**`pytorch-transformer/` is the project.** Everything tracked in git that
matters lives in `results/`.

**`model/`, `drive_bpe/`, `graphs/` and `tools/` are local-only.** They are
not pushed to GitHub: `model/` holds an 838 MB checkpoint, and the other
three are working copies of things that are either regenerable
(`graphs/`, via `tools/make_graphs.py`) or already captured in the tracked
results (`drive_bpe/`'s metrics CSV, which is also saved under
`results/bpe/`). They stay on disk so the project still runs end to end.

### `karpathy/` — what I want to learn next

Reference material, not part of this project.

- `minbpe/` — Andrej Karpathy's BPE written from scratch, in stages:
  `base.py`, `basic.py`, `gpt4.py`, `regex.py`, with `lecture.md` and tests.
  This is where I go to understand the tokenizer that beat mine, and why it
  would win with more data.
- `makemore_transformer.ipynb` — the decoder-only (GPT-style) transformer,
  built character by character. This is the one to learn next: it is a
  different animal from the encoder-decoder model in this project, since it
  has no encoder and no cross-attention.
- `names.txt` and `makemore.py` — the classic generative-model walkthrough.

Together with the `LLaMA/` papers (LLaMA 1, 2 and 3), the goal is to move
from the encoder-decoder Transformer to a decoder-only one, which is the
architecture behind every current large language model.

### `foundation/`

Two notebooks from before the transformer: `perceptron-xor-failure-and-solution.ipynb`
and `backpropagation_from_scratch.ipynb`. The perceptron failing on XOR and
then being fixed with a hidden layer is the reason the multi-head attention
and the position-wise feed-forward block are shaped the way they are — they
are hidden layers, and the XOR notebook is where that idea was earned.

---

## Running it

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate            # venv/bin/activate on macOS/Linux
pip install -r pytorch-transformer\requirements.txt

# 2. translate a sentence with the trained checkpoint
cd pytorch-transformer
python translate.py "But it was too late."
python translate.py --checkpoint 17 "The cat sleeps on the table."

# 3. train from scratch
#    NUM_EPOCHS defaults to 1 so a stray run cannot fill the disk.
#    Every knob below is an environment variable; see config.py.
set NUM_EPOCHS=20 & set EARLY_STOP_PATIENCE=2 & python train.py
```

`config.py` is the single source of truth for the run, and every value can be
overridden by an environment variable:

| Variable | Default | What it does |
|---|---|---|
| `NUM_EPOCHS` | `1` | how many epochs to train |
| `LR` | `1e-4` | learning rate |
| `LR_SCHEDULE` | `none` | `none`, `noam` (the paper's schedule), `cosine`, or `linear_decay` |
| `EARLY_STOP_PATIENCE` | `0` | stop after N epochs with no validation BLEU improvement (`0` = never) |
| `KEEP_BEST` | `1` | also save the best checkpoint as `tmodel_best.pt` |
| `SAVE_WEIGHTS_ONLY` | `0` | `1` drops the Adam moments: ~294 MB per checkpoint instead of ~838 MB |
| `VAL_SAMPLES` | `200` | how many validation sentences to decode and score each epoch |
| `DROPOUT`, `LABEL_SMOOTHING` | `0.1`, `0.1` | regularisation |

Per-epoch metrics are appended to `results/wordlevel/training_metrics.csv`,
so the peak epoch can always be identified after the fact.

The dataset (`Helsinki-NLP/opus_books`) is downloaded from Hugging Face on
first run. `colab_transformer_training.ipynb` is the same run on a Colab T4
with TensorBoard attached.

To reproduce the analysis in `results/`, the local `tools/` directory has one
script per question: scoring, the checkpoint sweep, the generalisation-gap
measurement, the source ablation, the side-by-side translations, the plots
and the attention heatmaps.

---

## Where it goes next

The architecture is learned and it is solid. What is missing is scale, and
scale is a data problem, not an architecture problem:

1. **More data.** 34,786 pairs is a demonstration set, not a training set. A
   million-plus pairs of `opus_books` or `NLLB` would be the single biggest
   lever, and it is the lever that would flip the BPE result.
2. **Turn on the learning-rate schedule.** The NoAM schedule and a cosine
   decay are already implemented in `config.py` (`LR_SCHEDULE=noam` or
   `cosine`), but the checkpoint on disk was trained with a constant LR, so
   they were never switched on. A warmup-plus-decay schedule is worth a
   couple of BLEU on its own.
3. **Turn on early stopping.** `EARLY_STOP_PATIENCE=2` would have stopped the
   run at epoch 18 instead of 19, and it keeps the best checkpoint rather
   than the last one. The run went 3 epochs past its own peak and threw away
   16% of the score. Stopping at the peak is free.
4. **Beam search with length normalisation.** Greedy decoding was used
   throughout for simplicity; beam search is a few points of BLEU for very
   little work.
5. **Fix the tokenizer leakage** in `train.py:333-334` so the tokenizers are
   built from the training split only.
6. **Then move to a decoder-only model** — the `karpathy/` material, and
   eventually a small LLaMA.
