# `training_metrics.csv` - provenance

The CSV that `train.py` writes had to be **reconstructed**, because the first
BPE run wrote it into the Colab runtime directory instead of Google Drive.

| rows | where they came from | how trustworthy |
|---|---|---|
| epoch 0-4 | downloaded from Drive - written by `train.py` during the run | original, untouched |
| epoch 5-9 | recovered from the run's own TensorBoard event files (`drive_bpe/transformers-from-scratch_runs_bpe-*.zip`) - the same scalars `train.py` logged | original values, just never synced |
| epoch 10 | train loss from TensorBoard; BLEU/CER/WER/p1-p4 from the offline evaluation of `tmodel_10.pt` on the same 200 seeded val sentences (`results/bpe/metrics_tmodel_10_n200.json`, `cased_as_logged` block) | equivalent protocol, but it is the offline number, not the in-training one |

`minutes` is blank for the reconstructed rows: the wall-clock time per epoch was
not in TensorBoard, and inventing it would be a lie. Rows 0-4 show ~27.6-27.8
min/epoch on a Colab T4.

The reason the CSV was lost: `config.py` pointed `metrics_file` at
`results/<variant>/training_metrics.csv`, which is inside the ephemeral Colab
runtime. Fixed in `config.py` + `train.py` - the CSV is now written next to
the checkpoints, i.e. on Drive.

## The run in one table

| epoch | global_step | train loss | BLEU-4 | CER | WER |
|---|---|---|---|---|---|
| 0 | 4,349 | 6.0576 | 1.61 | 0.849 | 1.249 |
| 1 | 8,698 | 5.3503 | 2.51 | 0.887 | 1.246 |
| 2 | 13,047 | 4.9634 | 3.93 | 0.712 | 1.056 |
| 3 | 17,396 | 4.6695 | 4.73 | 0.742 | 1.100 |
| 4 | 21,745 | 4.4201 | 6.18 | 0.698 | 1.012 |
| 5 | 26,094 | 4.1814 | 6.21 | 0.742 | 1.073 |
| 6 | 30,443 | 3.9648 | 7.27 | 0.682 | 0.987 |
| 7 | 34,792 | 3.7634 | 7.57 | 0.720 | 1.017 |
| 8 | 39,141 | 3.5801 | 8.18 | 0.672 | 0.960 |
| 9 | 43,490 | 3.4033 | 8.60 | 0.668 | 0.943 |
| 10 | 47,839 | 3.3892 | 8.32 | 0.699 | 0.996 |

BLEU-4 is the cased number (as logged). The BPE tokenizer lowercases its
output while the references stay cased, so a case-blind score of the final
checkpoint is **11.01** - see `results/bpe/metrics_tmodel_10_n200.json`.
