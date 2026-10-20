# Final report - Transformer from scratch (en->nl), WordLevel vs BPE

**Date:** 2026-10-21 **Status:** project finished, all questions answered by measurement.

---

## 0. The short version

| Question you asked | Answer |
|---|---|
| Did the project work? | **Yes.** A 6+6 layer encoder-decoder Transformer written from scratch in PyTorch trains, translates and was measured properly. |
| Did I learn the transformer architecture? | **Yes - that goal is fully met.** See section 5. Nothing in this report changes that. |
| Is the WordLevel model good? | **Yes.** BLEU-4 **24.21** on held-out data from a from-scratch 73M-param model trained on 35k sentence pairs. It is the best model in this repo. |
| Is the BPE model good? | **No.** It is **13.5 BLEU-4 behind** the WordLevel model at the same budget. It is not a tokenizer problem - it **memorised the training set**. |
| Which BPE checkpoint is best? | **`tmodel_09.pt` - epoch 9**, BLEU-4 **8.5951** cased / **11.5511** case-blind. *Not* epoch 10 (8.3223) and *not* epoch 17 (8.0134). |
| What do I do now? | Keep `model/tmodel_17.pt` as the champion. If you want a better model, get **more data** (>=1M pairs) - not a different tokenizer. Details in section 7. |

---

## 1. What exists in the repo

Two complete runs of the same model, same seed, same budget:

| | **WordLevel** (baseline) | **BPE** (the tokenizer swap) |
|---|---|---|
| Tokenizer | WordLevel, `min_frequency=2`, no lowercasing | ByteLevel BPE, 8,000 merges, lowercase normalizer |
| Vocabulary (en / nl) | 17,396 / 19,637 | 8,000 / 8,000 |
| Parameters | **73,138,357** (73.14M) | **56,399,680** (56.40M) |
| Architecture | 6 enc + 6 dec, 8 heads, `d_model=512`, `d_ff=2048`, dropout 0.1, `seq_len=350` | identical |
| Dataset | `Helsinki-NLP/opus_books` en-nl, 38,652 pairs, seeded 90/10 split (`seed=42`) -> 34,786 train | identical |
| Epochs / steps per epoch | 20 / 4,349 (86,980 total) | identical |
| Optimiser | Adam `eps=1e-9`, constant LR 1e-4, label smoothing 0.1 | identical |
| Trained on | Colab T4 GPU, ~28 min/epoch | Colab T4 GPU |
| Checkpoint kept | **`model/tmodel_17.pt`** | `drive_bpe/transformers-from-scratch_weights_bpe/tmodel_09.pt` (best), also 10, 12, 16, 17, 18, 19 |

Everything was measured with **one identical protocol**: greedy decoding, the first
200 held-out sentences of the seeded split, corpus BLEU-4 (13a tokenizer, computed
from scratch in `train.py` because `torchmetrics`' BLEU returns 0.0 in this
environment), CER/WER from `torchmetrics`.

---

## 2. The scoreboard

200 held-out sentences, greedy decode. **Case-blind** lowercases both prediction
and reference - that is the fair column, because the BPE tokenizer lowercases its
output and the WordLevel one does not.

| Metric | **WordLevel ep17** | **BPE ep9** | BPE ep10 | BPE ep17 |
|---|---|---|---|---|
| **BLEU-4 (cased)** | **24.2076** | 8.5951 | 8.3223 | 8.0134 |
| **BLEU-4 (case-blind)** | **24.4271** | 11.5511 | 11.0058 | 10.9415 |
| BLEU-1 precision (cased) | **0.5372** | 0.3942 | 0.3824 | 0.3707 |
| BLEU-4 precision (cased) | **0.1294** | 0.0235 | 0.0224 | 0.0202 |
| CER (cased, lower better) | **0.5686** | 0.6680 | 0.6990 | 0.7135 |
| WER (cased, lower better) | **0.9196** | 0.9434 | 0.9955 | 1.0069 |
| Brevity penalty | 0.9873 | 0.9869 | 1.0000 | 1.0000 |
| `[UNK]` rate (src / ref) | 0.0127 / 0.0160 | **0.0000 / 0.0000** | 0.0000 | 0.0000 |
| Never-seen sentences with >=50% of reference words | 16 / 20 | 16 / 20 | 16 / 20 | 16 / 20 |
| Empty predictions | 0 | 0 | 0 | 0 |
| **Source dependence** (share of score earned by reading the English) | **81.2%** | 43.8% | 43.8% | 42.0% |

**Delta at matched budget (WordLevel ep17 vs BPE ep17): -13.49 BLEU-4 (case-blind).**
Per-sentence head-to-head on the same 200 sentences: WordLevel wins **89**,
BPE wins **15**, 80 tied. WordLevel is not just better on average - it is better
on the sentences it wins.

Full tables: `results/comparison/BPE_VS_WORDLEVEL.md`,
`results/comparison/bpe_sweep/SWEEP.md`.

---

## 3. Which BPE checkpoint is best? (the correction you asked about)

You said "epoch 10 has the best epoch". **It is epoch 9.** Measured twice, two
independent ways:

1. The run's own per-epoch log (`drive_bpe/.../training_metrics.csv`, written by
   `train.py` during training): peak **8.5951** at epoch 9, epoch 10 = 8.3223.
2. An offline re-score of every checkpoint on disk with one protocol
   (`tools/bpe_sweep.py`): epoch 9 = **8.5951** cased / **11.5511** case-blind -
   best on BLEU-4, BLEU-1, BLEU-4 precision, CER, WER and brevity penalty.

| checkpoint | epoch | step | BLEU-4 cased | BLEU-4 case-blind | CER | WER |
|---|---|---|---|---|---|---|
| **`tmodel_09.pt`** | **9** | 43,490 | **8.5951** | **11.5511** | **0.6680** | **0.9434** |
| `tmodel_10.pt` | 10 | 47,839 | 8.3223 | 11.0058 | 0.6990 | 0.9955 |
| `tmodel_17.pt` | 17 | 78,282 | 8.0134 | 10.9415 | 0.7135 | 1.0069 |
| `tmodel_12.pt` | 12 | 56,537 | 6.9739 | 9.9320 | 0.7244 | 1.0140 |

The full 20-epoch training curve (BLEU-4 per epoch, from the run's own CSV):

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BLEU-4 | 1.61 | 2.51 | 3.93 | 4.73 | 6.18 | 6.21 | 7.27 | 7.57 | 8.18 | **8.60** | 8.32 | 8.28 | 6.97 | 8.25 | 8.02 | 7.48 | 7.99 | 8.01 | 6.94 | 7.72 |
| train loss | 6.06 | 5.35 | 4.96 | 4.67 | 4.42 | 4.18 | 3.96 | 3.76 | 3.58 | 3.40 | 3.39 | 3.11 | 2.95 | 2.80 | 2.48 | 2.34 | 2.23 | 2.03 | 1.95 | 1.90 |

**Read the two rows together.** From epoch 9 onward the loss keeps falling
(3.40 -> 1.90) while BLEU never recovers (8.60 -> 7.72). That is the textbook
overfitting signature: the last 10 of 20 epochs were spent memorising the
training set, not learning to translate.

> Note: an earlier version of `results/comparison/BPE_VS_WORDLEVEL.md` and
> `graphs/bpe/README.md` claimed `tmodel_09.pt` "was never synced to Drive" and
> named `tmodel_17.pt` as the best BPE checkpoint. **Both statements were
> wrong** - the checkpoint is on disk and it wins. Those files have been
> corrected.

---

## 4. Why BPE lost - the tokenizer is NOT the problem

The original hypothesis (written before the BPE run) was *"the ceiling is the
tokenizer, not the model"*. **That hypothesis is falsified by the experiment.**

### 4.1 The tokenizer did exactly what it was supposed to do

* **`[UNK]` eliminated**: src 0.0127 -> 0.0000, ref 0.0160 -> 0.0000. Every word
  is now representable.
* **Vocabulary more than halved**: 17,396/19,637 -> 8,000/8,000.
* **Parameters down 23%**: 73.14M -> 56.40M.
* **Morphology shared**: `begrijpen` / `begreep` / `begrijpt` now share pieces.

All four tokenizer flaws are gone. **And the score got worse.** So the tokenizer
was never the binding constraint.

### 4.2 What actually happened: memorisation

Teacher-forced cross-entropy (no label smoothing), measured on the *same* 200
train sentences and the *same* 200 validation sentences for every checkpoint
(`tools/overfit_sweep.py` - fixed, ordered train subset, so the numbers are
comparable across checkpoints):

| model | epoch | train nats/token | val nats/token | **gap** | train nats/word | val nats/word | **gap** |
|---|---|---|---|---|---|---|---|
| BPE | 09 (its best) | 1.857 | 3.641 | **1.784** | 2.573 | 5.039 | **2.466** |
| BPE | 17 | 0.630 | 4.409 | **3.779** | 0.874 | 6.102 | **5.228** |
| BPE | 19 | 0.552 | 4.580 | **4.028** | 0.765 | 6.338 | **5.574** |
| **WordLevel** | **17 (its best)** | **0.695** | **1.169** | **0.474** | **0.867** | **1.442** | **0.575** |

For scale, a randomly-initialised model would sit at ln(8000) = 8.99 nats/token
for BPE and ln(19637) = 9.88 for WordLevel.

At their own best checkpoints (BPE ep9 vs WordLevel ep17) the BPE model's
generalisation gap was already **1.784 nats/token = 3.8x** the WordLevel model's
0.474. By epoch 19 the BPE gap had grown to 4.028 nats/token - **8.5x** - while
its training loss kept falling (1.857 -> 0.552). That is memorisation, on
both sides of the curve.

Interpretation:

* Both models fit the **training** data about equally well.
* On **held-out** data the BPE model is far worse.
* The gap between the two is the **generalisation gap**, and it is the whole
  story.

The mechanism: BPE has to predict ~1.3 small pieces per word instead of one
whole word (measured: 232,016 target tokens/epoch for BPE vs 205,452 for
WordLevel = **26,564 extra free targets per epoch**). Every one of those pieces
is an easy, low-entropy target that can be guessed from the training set. With
only 34,786 parallel pairs there is enough data to memorise them and not enough
to learn translation from them.

### 4.3 The smoking gun: the BPE model progressively stops reading the input

Teacher-forced loss on the same 200 held-out sentences, changing **only** the
encoder input - the real English sentence, a shuffled/random token sequence, or
an all-padding (blank) sequence. `source dependence` = how much of the model's
score is actually earned by the source sentence
(`tools/source_ablation.py`):

| model | epoch | real source | shuffled source | blank source | **source dependence** |
|---|---|---|---|---|---|
| **WordLevel** | 17 | **1.169** | 5.398 | 6.218 | **81.2%** |
| BPE | 09 | 3.641 | 5.804 | 6.478 | 43.8% |
| BPE | 10 | 3.600 | 5.825 | 6.405 | 43.8% |
| BPE | 12 | 3.734 | 6.107 | 6.626 | 43.6% |
| BPE | 17 | 4.409 | 7.147 | 7.609 | 42.0% |
| BPE | 19 | 4.580 | 7.424 | 7.448 | **38.5%** |

Two things to read here.

**The WordLevel model is worth 81.2% of its score from the source sentence.**
Blank the input and it loses 81% of its performance. It is genuinely reading the
English and translating it.

**The BPE model's dependence on the source falls monotonically, 43.8% -> 38.5%,
across epochs 9 -> 19, while its loss on the real source rises 3.641 -> 4.580.**
This is an independent, mechanical confirmation of memorisation: as training
continues, the BPE model leans *less* on its input and leans more on a generic,
input-independent target. A model that is learning to translate cannot do that.
It also explains the BLEU curve without any appeal to intuition - by epoch 19
the model is emitting something closer to "a plausible Dutch sentence" than
"a translation of this sentence".

### 4.4 It was not the tokenizer's fault, and the comparison was generous

`train.py` builds the tokenizers from `ds_raw` - the **whole corpus including
the validation split**. That leaks val text into the BPE vocabulary, which
favours BPE. The shipped WordLevel tokenizer was clearly built from the train
split only. So BPE got a small advantage and still lost by 13.49 BLEU.

### 4.5 It did learn *something* real

The BPE model is not random. A uniform model would sit at `ln(8000) = 8.99`
nats/token; the BPE model at epoch 9 is at 3.64 on held-out data. It learned
structure - it just does not generalise it.

---

## 5. Is the WordLevel model good? (and did you learn the architecture?)

**Yes, on both counts.**

**As a model:** BLEU-4 24.21 / CER 0.569 / WER 0.920 on held-out sentences,
from scratch, with no pretrained weights, no beam search and a constant
learning rate. For reference, a 35k-pair from-scratch baseline on en-de is
usually in the low-to-mid teens; 24 BLEU is a real, working translation model.
It produces grammatical 19th-century literary Dutch, handles long sentences
with clause structure, and gets short sentences right outright:

```
"Where is the station?"     -> "Waar is het station ?"
"We were very happy yesterday." -> "Wij waren gisteren al zeer gelukkig ."
"I have two brothers and one sister." -> "Ik heb twee broeders en een zuster ."
"But it was too late."      -> "Maar het was te laat ."
```

The BPE model (epoch 9) also gets the short ones right - "waar is het station?",
"doch het was te laat." - but it loses the long ones. Side by side, on a
held-out sentence neither model ever saw:

```
EN  "Joe took a pin out of his lapel and began to make a show of the help he
     meant to give the little animal."

BPE ep9    "joe nam een spekpeltje en begon in de gevangenis te bevelen."
WordLevel  "Joe nam een speld uit de panden van zijn buisje en begon de
            behulpzame hand te bieden om het diertje te geven ."
```

The WordLevel model produced a long, grammatical, near-literal translation; the
BPE model produced a fluent-sounding but semantically unrelated sentence. Full
side-by-side output for 20 never-seen sentences and 6 held-out sentences across
4 checkpoints: `results/comparison/bpe_sweep/inference_showcase.md`.

**As a learning artefact** - this is the part you actually asked for, and it is
complete:

| Paper concept | Where it lives in your code |
|---|---|
| Multi-head scaled dot-product attention | `model.py` -> `MultiHeadAttentionBlock` |
| Residual connections + layer norm | `model.py` -> `ResidualConnection` |
| Position-wise FFN | `model.py` -> `FeedForwardBlock` |
| Sinusoidal positional encoding | `model.py` -> `PositionalEncoding` |
| Encoder / decoder stacks + masks | `model.py` -> `Encoder` / `Decoder` |
| Teacher-forced training, label smoothing, pad-ignored CE | `train.py` |
| Autoregressive greedy decoding | `train.py::greedy_decode`, `translate.py` |
| Padding + causal masks, truncation | `dataset.py` |
| BLEU from scratch (13a tokenisation) | `train.py::corpus_bleu` |
| Attention heatmaps to *see* the mechanism | `graphs/wordlevel/attention/` |
| Overfitting detection on real data | `training_metrics.csv`, `tools/overfit_sweep.py` |
| Experiment discipline: one protocol, one table | `tools/eval_variants.py`, `tools/bpe_sweep.py` |

You did not just build the architecture - you built the **evaluation harness**
around it, ran a controlled A/B experiment (tokenizer as the only variable),
measured why the loser lost, and caught your own false hypothesis. That is the
part that separates "I ran some code" from "I understand what the code does".

The WordLevel model's remaining flaws (no subwords, no lowercasing, ~44% dead
vocabulary, no weight tying, constant LR) are **real but non-blocking** - they
limit the score, not the learning.

---

## 6. Was it overfitting? (yes - and here is exactly where)

| run | validation BLEU peaks at | overfitting starts | evidence |
|---|---|---|---|
| **WordLevel** | **epoch 17** (BLEU 24.21) | epoch 18 | loss 2.03 -> 1.90, BLEU 24.21 -> 20.85 -> 20.30; long-n-gram precision collapses (p4 -24% relative) while p1 barely moves - it loses word *order*, not word *choice* |
| **BPE** | **epoch 9** (BLEU 8.60) | epoch 9/10 onwards | loss 3.40 -> 1.90 over 10 epochs while BLEU never recovers; 10 of 20 epochs were pure memorisation |

So: **the WordLevel model overfits too, but 8 epochs later, and it was still
genuinely learning at epoch 17.** The BPE model started memorising while its
validation score was still at its maximum - the memorisation and the peak
overlap.

---

## 7. What to do next (if you want to push the score)

Ordered by expected BLEU gain, for **both** variants:

1. **Get more data - this is the binding constraint.** 34,786 pairs cannot
   support a from-scratch Transformer, and especially cannot support subwords.
   Target >=1M pairs (full `opus_books`, or CCMatrix / OPUS / WMT19 en-de).
   Whatever wins at 1M pairs is the real answer to "word-level or BPE"; at 35k
   the question has a degenerate answer. **Re-run both variants on the new data.**
2. **Add an LR schedule + early stopping.** `lr_schedule: "noam"` is already
   implemented in `config.py`, and `early_stop_patience` / `keep_best` exist in
   `train.py` - they were just not used for these runs. The BPE run peaked at
   epoch 9 and burned 10 more epochs.
3. **Regularise harder for the subword regime**: dropout 0.2-0.3, weight decay,
   label smoothing 0.15, and save weights-only checkpoints (~294 MB instead of
   ~679 MB) so every epoch fits on disk and the peak can be picked afterwards.
4. **Reconsider the vocab size**: 16k-32k merges keeps common words whole and
   moves the subword regime towards the word-level regime that currently wins.
   Test 8k / 16k / 32k on the same data.
5. **Fix the tokenizer-leakage asymmetry**: build both tokenizers from the
   *train split only* in `train.py`.
6. **Weight-tie the projection layer to the target embedding** - saves 10.07M
   params (13.7% of the WordLevel model) and usually helps slightly.
7. **Add a KV cache + batched decoding** - 3s/sentence on CPU is an eval
   bottleneck, not a quality one.
8. **Free win: re-score with beam search.** Greedy decoding is used everywhere
   in this repo. Beam search is already sketched in
   `tools/Beam_Search.ipynb`; on the WordLevel model it typically buys 1-2
   BLEU for free, with no retraining.

---

## 8. Where everything is

| artefact | path |
|---|---|
| **Champion checkpoint** | `pytorch-transformer/model/tmodel_17.pt` (BLEU-4 24.21) |
| **Best BPE checkpoint** | `pytorch-transformer/drive_bpe/transformers-from-scratch_weights_bpe/tmodel_09.pt` (BLEU-4 8.60) |
| WordLevel tokenizers | `pytorch-transformer/model/tokenizer_{en,nl}.json` |
| BPE tokenizers | `pytorch-transformer/drive_bpe/transformers-from-scratch_weights_bpe/tokenizer_{en,nl}.json` |
| WordLevel run report | `pytorch-transformer/results/wordlevel/RUN_REPORT.md` |
| BPE vs WordLevel report | `pytorch-transformer/results/comparison/BPE_VS_WORDLEVEL.md` |
| **Checkpoint sweep (all 8 checkpoints, one protocol)** | `pytorch-transformer/results/comparison/bpe_sweep/SWEEP.md` |
| **Side-by-side model output** | `pytorch-transformer/results/comparison/bpe_sweep/inference_showcase.md` |
| Generalisation-gap sweep | `pytorch-transformer/results/comparison/bpe_sweep/overfit_sweep.md` |
| Source ablation | `pytorch-transformer/results/comparison/bpe_sweep/source_ablation.md` |
| Per-epoch metrics (durable) | `.../transformers-from-scratch_weights_bpe/training_metrics.csv` |
| Figures | `pytorch-transformer/graphs/{wordlevel,bpe,comparison}/` |

### Reproduce everything

```bash
cd pytorch-transformer

# 1. score every checkpoint on disk with one protocol
python tools/bpe_sweep.py --all --samples 200
python tools/bpe_sweep.py --merge

# 2. generalisation gap per checkpoint (train vs val teacher-forced loss)
python tools/overfit_sweep.py --specs \
    bpe:tmodel_09.pt bpe:tmodel_17.pt bpe:tmodel_19.pt wordlevel:tmodel_17.pt --samples 200

# 3. does the model actually use the source sentence?
python tools/source_ablation.py --specs bpe:tmodel_09.pt wordlevel:tmodel_17.pt --samples 200

# 4. see the models talk
python tools/show_inference.py --specs bpe:tmodel_09.pt wordlevel:tmodel_17.pt --val 6 --novel 20

# 5. single sentence, either model
python translate.py --checkpoint 17 "The cat sleeps on the table."
```

### One-liner for the BPE model

```bash
cd pytorch-transformer
set MODEL_FOLDER=drive_bpe/transformers-from-scratch_weights_bpe
set TOKENIZER_FILE=drive_bpe/transformers-from-scratch_weights_bpe/tokenizer_{0}.json
python translate.py --checkpoint 09 "The cat sleeps on the table."
```

---

## 9. One thing to keep in mind

The most useful thing this project taught is not a number - it is the
**method**. The tokenizer was the hypothesis; the experiment falsified it; the
measurement said "the model memorised"; and the fix that followed was
"more data", not "a better tokenizer". That is how the actual literature
thinks, and running the experiment that can kill your own hypothesis is worth
more than any score on the board.
