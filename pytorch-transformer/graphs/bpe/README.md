# BPE run - not started yet

This directory is reserved for the **BPE / subword** tokenizer run, so its graphs
never mix with the WordLevel baseline in `../wordlevel/`.

## Why this run matters

The WordLevel baseline (`../wordlevel/`) plateaus at **BLEU-4 24.21** and the
ceiling is the tokenizer, not the model. From `../../wordlevel_epoch16_samples.md`:

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

## The comparison to beat

| | WordLevel baseline | BPE (target) |
|---|---|---|
| Checkpoint | `model/tmodel_17.pt` | _not trained_ |
| BLEU-4 | 24.21 | ? |
| CER | 0.569 | ? |
| WER | 0.920 | ? |
| Vocab (en / nl) | 17,396 / 19,637 | expect far smaller |
| Params | 73.50M (18.96M embeddings) | ? |

Keep the training budget comparable (same 20 epochs, same seed, same
`seq_len`) or the comparison is not meaningful. Epoch 17 is the number that
matters, not epoch 19 - the WordLevel run was already overfitting by then.
