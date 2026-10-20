# BPE run - trained, evaluated, and it LOST

This directory is reserved for the **BPE / subword** tokenizer run, so its graphs
never mix with the WordLevel baseline in `../wordlevel/`.

> **Status:** the run finished (20 epochs, seed 42, same `seq_len` 350, same
> 34,786 training pairs as the baseline) and was evaluated with the identical
> protocol. Full report: `../../results/comparison/BPE_VS_WORDLEVEL.md`.

## Why this run matters

The WordLevel baseline (`../wordlevel/`) plateaus at **BLEU-4 24.21** and the
hypothesis was that the ceiling is the tokenizer, not the model. From
`../../wordlevel_epoch16_samples.md`:

- ~44% of the English vocabulary and ~40% of the Dutch vocabulary are dead entries
- any word seen once becomes `[UNK]` (`min_frequency=2` discards it)
- no subwords, so `lopende` / `lopend` / `gelopen` share no representation
- no lowercasing, so `zij` and `Zij` are separate entries (981 such words)

Switching to a BPE tokenizer should attack all four at once.

## What to do here

1. Add a `bpe` tokenizer option to `get_or_build_tokenizer()` in `train.py`
   (keep the WordLevel path so the baseline stays reproducible).
2. Lowercase before tokenizing, and normalise the punctuation noise
   (`--„`, `„‚`, `,--„` are real vocabulary entries in the WordLevel run).
3. Train a fresh model - do **not** resume from `tmodel_17.pt`. The embedding
   matrix and projection layer are sized to the old vocabulary and will not load.
4. Write the event logs somewhere separate, e.g.
   `drive/transformers-from-scratch_runs_bpe/tmodel_en_nl/`, then point
   `RUNS_DIR` in `make_graphs.py` at them.
5. Score it against the same protocol so the numbers are comparable:
   `python evaluate.py --samples 200`

Steps 1-5 are done: the `bpe` tokenizer option lives in `train.py`, the
tokenizers were trained with a Lowercase normalizer and saved next to the
checkpoints in `drive_bpe/transformers-from-scratch_weights_bpe/`, and the run
was scored with `tools/eval_variants.py` (the same protocol as the baseline).

## The comparison to beat - RESULT

| | WordLevel baseline | BPE (target) | BPE (actual, ep17) |
|---|---|---|---|
| Checkpoint | `model/tmodel_17.pt` | - | `drive_bpe/transformers-from-scratch_weights_bpe/tmodel_17.pt` |
| BLEU-4 (cased) | 24.21 | ? | **7.99** |
| BLEU-4 (case-blind) | 24.43 | ? | **10.94** |
| CER | 0.569 | ? | **0.714** |
| WER | 0.920 | ? | **1.007** |
| Vocab (en / nl) | 17,396 / 19,637 | expect far smaller | **8,000 / 8,000** |
| Params | 73.50M (18.96M embeddings) | ? | **56.40M** |
| `[UNK]` rate (src / ref) | 0.0127 / 0.0160 | ? | **0.0000 / 0.0000** |

Keep the training budget comparable (same 20 epochs, same seed, same
`seq_len`) or the comparison is not meaningful. Epoch 17 is the number that
matters, not epoch 19 - the WordLevel run was already overfitting by then.

### Verdict

The BPE model is **13.49 BLEU-4 below** the WordLevel model at the matched
budget (24.43 vs 10.94 case-blind). The tokenizer itself worked exactly as
intended - `[UNK]` is gone, the vocab is less than half the size, the model is
16.74M parameters smaller - but the score still collapsed.

The reason is measured, not guessed: the BPE model **memorised the training
set**. Teacher-forced loss on 200 train vs 200 val sentences:

| | train nats/token | val nats/token | gap |
|---|---|---|---|
| WordLevel ep17 | 0.797 | 1.169 | 0.372 |
| BPE ep17 | 0.738 | 4.409 | 3.671 |

Both models fit the training data equally well; BPE is 3.8x worse on held-out
data. The hypothesis above ("the ceiling is the tokenizer, not the model") is
therefore **falsified** - the binding constraints are the amount of data
(34,786 pairs) and the optimisation schedule (constant LR, no early stopping),
not the tokenizer.

Also worth knowing: the BPE run's own peak was **epoch 9 (BLEU-4 8.60)**. That
checkpoint (`tmodel_09.pt`) **is** on disk now - it was synced to Drive after
this file was first written - and an offline re-score of every checkpoint on
disk confirms it is the best one:

| checkpoint | BLEU-4 cased | BLEU-4 case-blind | CER | WER |
|---|---|---|---|---|
| **`tmodel_09.pt`** (peak) | **8.5951** | **11.5511** | **0.6680** | **0.9434** |
| `tmodel_10.pt` | 8.3223 | 11.0058 | 0.6990 | 0.9955 |
| `tmodel_17.pt` | 8.0134 | 10.9415 | 0.7135 | 1.0069 |
| `tmodel_12.pt` | 6.9739 | 9.9320 | 0.7244 | 1.0140 |

So the best BPE checkpoint in the repo is **`tmodel_09.pt`**, not `tmodel_17.pt`
- epoch 17 is the *matched-budget* comparison point, not the best one.
Sweep: `results/comparison/bpe_sweep/SWEEP.md`.

Next step is not another tokenizer: it is more data (>= 1M pairs) plus an LR
schedule and early stopping, re-run for *both* variants.

## Figures for this run

| file | what it shows |
|---|---|
| `train_loss.png` | per-step training loss, 20 epochs |
| `val_bleu.png` | validation BLEU-4 per epoch, with the peak marked |
| `val_error_rates.png` | validation CER / WER per epoch |
| `bleu_precisions.png` | BLEU-1..4 precisions per epoch |
| `training_dashboard.png` | all four in one 2x2 dashboard |
| `overfitting.png` | validation BLEU next to the training loss - the moment memorising starts |
| `../comparison/val_bleu_vs_epochs.png` | WordLevel vs BPE validation BLEU |
| `../comparison/overfitting_vs_epochs.png` | where each run starts overfitting |
| `../comparison/memorisation_evidence.png` | teacher-forced train vs val loss for both checkpoints |

Regenerate them with:

```bash
python tools/make_graphs.py --variant bpe
python tools/make_graphs.py --variant wordlevel
python tools/make_comparison_graphs.py
```

