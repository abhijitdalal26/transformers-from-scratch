# WordLevel run report - tmodel

Everything measured for the English→Dutch WordLevel run, and why epoch 17 is the
checkpoint to keep.

- **Model:** `model/tmodel_17.pt` (epoch 17, global_step 78,282)
- **Best score:** BLEU-4 **24.21**, CER 0.569, WER 0.920 on 200 held-out sentences
- **Graphs:** `../../graphs/wordlevel/`
- **Supersedes:** `../../wordlevel_epoch16_samples.md` (kept as the epoch-16
  vocabulary/tokenizer deep-dive that this report builds on)

---

## 1. Setup

| | |
|---|---|
| Task | English → Dutch machine translation |
| Dataset | `Helsinki-NLP/opus_books` — 38,652 pairs (19th-c. literary prose) |
| Split | 90 / 10, **seeded** (`seed=42`) → 34,786 train / 3,866 val |
| Tokenizer | WordLevel, `Whitespace` pre-tokenizer, **no normalizer**, `min_frequency=2` |
| Vocabulary | 17,396 (en) / 19,637 (nl) |
| Architecture | 6+6 layers, 8 heads, `d_model=512`, `d_ff=2048`, dropout 0.1, `seq_len=350` |
| Parameters | **73.50M** total, **18.96M** embeddings (25.8%) |
| Optimiser | Adam, `eps=1e-9`, **constant** LR 1e-4 (no schedule) |
| Loss | cross-entropy, `label_smoothing=0.1`, target `[PAD]` ignored |
| Training | 20 epochs (0–19), 86,980 steps, batch 8, Colab T4 |

Parameter split, measured not estimated:

| Component | Parameters | Share |
|---|---|---|
| Source embedding | 8.91M | 12.1% |
| Target embedding | 10.05M | 13.7% |
| Encoder (6 blocks) | 18.90M | 25.7% |
| Decoder (6 blocks) | 25.20M | 34.3% |
| Projection layer | 10.07M | 13.7% |
| **Total** | **73.50M** | 100% |

The projection layer is a separate 19,637 × 512 matrix and is **not weight-tied**
to the target embedding, so the target vocabulary is paid for twice
(10.05M + 10.07M = 20.1M, 27.4% of the model).

---

## 2. Results

Both checkpoints scored offline on the **same** 200 held-out sentences, greedy
decoding, seeded split. The numbers reproduce the training-time TensorBoard values
to 3 decimal places, which confirms the split is stable.

| Metric | **`tmodel_17`** | `tmodel_19` (last) | Δ |
|---|---|---|---|
| **BLEU-4** | **24.21** | 20.30 | **+3.91** |
| BLEU-1 precision | 0.5372 | 0.5160 | +0.021 |
| BLEU-2 precision | 0.2850 | 0.2493 | +0.036 |
| BLEU-3 precision | 0.1824 | 0.1481 | +0.034 |
| BLEU-4 precision | 0.1294 | 0.0988 | **+0.031** |
| Brevity penalty | 0.9873 | 0.9747 | +0.013 |
| CER (lower better) | **0.5686** | 0.5910 | −0.022 |
| WER (lower better) | **0.9196** | 0.9367 | −0.017 |
| Mean unigram recall | **0.5572** | 0.5219 | +0.035 |
| Mean generated length | 24.77 | 24.52 | +0.25 |
| Exact match | 0 | 0 | — |

Training loss fell from **9.9274** (≈ `ln(19637)` = 9.885, i.e. uniform
prediction) to **1.8316**.

A 60-sentence three-way head-to-head on identical sentences
(`compare_tmodel_17_vs_tmodel_18_vs_tmodel_19.md`) confirms it: corpus BLEU
**26.36 / 21.83 / 21.12**, and `tmodel_17` had the best per-sentence BLEU on
**38 of 60** against 11 each for epochs 18 and 19. It wins **11 of the 12**
directed metrics — the single exception is WER, where `tmodel_18` edges it
(0.900 vs 0.905) by being marginally too short rather than by being more correct.

---

## 3. Overfitting: it starts at epoch 18

**Answer: the peak is epoch 17, and epoch 18 is the first overfit epoch.**

| Epoch | global_step | BLEU-4 | CER | WER | Train loss |
|---|---|---|---|---|---|
| 17 | 78,282 | **24.21** | 0.569 | 0.920 | ~2.5 (falling) |
| 18 | 82,631 | 20.85 | 0.580 | 0.921 | falling |
| 19 | 86,980 | 20.30 | 0.591 | 0.937 | 1.83 |

BLEU dropped **3.91 points (16% relative)** over the last two epochs while training
loss kept falling. The last two epochs were wasted compute.

### The mechanism is visible in the precisions

| Epoch | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| 17 | 0.5372 | 0.2850 | **0.1824** | **0.1294** |
| 18 | 0.5210 | 0.2630 | 0.1570 | 0.1050 |
| 19 | 0.5160 | 0.2493 | 0.1481 | 0.0988 |

Short n-grams degrade gently (p1 −2.1 points) while long n-grams collapse
(p4 −3.1 points, −24% relative). The model kept getting *word choice* right but
lost *word order*. Generated length stayed flat (24.77 → 24.52), so it is not
drifting toward short or long output — it is genuinely getting worse at structure.
That is the signature of memorisation, not of a bad learning-rate tail.

### And it is visible in the text

From `compare_tmodel_17_vs_tmodel_19.md`, on a long sentence:

> **SOURCE:** About this time the two Miss Steeles ... presented themselves again
> before their more grand relations in Conduit and Berkeley Streets; and were
> welcomed by them all with great cordiality.
>
> **`tmodel_17`:** Omstreeks dezen tijd kwamen de dames Steele , die reeds een poosje
> waren gelogeerd bij hun neef in Bartlett's Buildings , Holborn , zich weer vertoonen
> bij hun verwanten in Conduit en Berkeley Street ; en zij werden door alle mogelijke
> vriendelijkheid aan hen ontvangen .
>
> **`tmodel_19`:** Omstreeks dezen tijd kwamen de beide dames Steele uit hun neef ,
> die te _Ami?ns_ hadden ontvangen , zich van de , reeds bezig hielden in hun en in
> Berkeley Street ; en werden reeds de kennis gegeven door al de van hen .

Epoch 17 tracks the clause structure faithfully. Epoch 19 degenerates. A second
case shows pure repetition collapse — `Ik houd niet van mijn , , , , dat zij zwaar
zijn`. Short sentences stay roughly comparable between the two; it is long
sentences where the overfitting bites.

### Caveat: the log cannot pin the peak any more precisely

Only **three** epochs have a valid BLEU value (17, 18, 19). Epochs 1–10 were
logged with the broken `torchmetrics` BLEU that always returned 0.0, and the
TensorBoard event files are missing steps 13,186–26,094 and 43,489–73,933 — which
covers roughly epochs 3–5 and 10–16. Only 46,995 of 86,980 steps survived (54%),
and three of the twelve event files contain nothing at all: the sessions wrote
to them but Colab never synced them, so those steps are gone rather than
corrupted. The true peak could be earlier than 17 and was never recorded. Two of
20 validation points are trustworthy.

The shaded gaps in `../../graphs/wordlevel/train_loss.png` mark this. A future
run cannot repeat it: `train.py` now writes `training_metrics.csv` as well, a
plain file that a disconnect cannot take away.

---

## 4. Why the score is capped: the tokenizer

The model is not weak — the tokenizer is. From
`../../wordlevel_epoch16_samples.md`:

- **44% of the English vocabulary and 40% of the Dutch vocabulary are dead
  entries** that never appear in the corpus.
- `min_frequency=2` means any word seen exactly once is discarded and becomes
  `[UNK]`. `[UNK]` rate: 0.60% of English tokens, 1.34% of Dutch.
- **No subwords.** `lopende` / `lopend` / `gelopen` share no representation, and
  the model has to learn each form from scratch.
- **No lowercasing.** `zij` and `Zij` are separate entries; 981 words are stored
  in both cases.
- 9 vocabulary entries are OCR/dialogue artefacts (`--„`, `„‚`, `,--„`).

This shows up directly in the generalisation test below: the model produces
`Ik drinken iedere .` for *"I drink coffee every morning"* — `drink` (correct,
1st person) and `drinken` (infinitive) are unrelated tokens with no shared
representation, so it picked the wrong one and lost the noun.

---

## 5. Generalisation: sentences never seen in training

The validation split is held out, but it is still the same book corpus. The 20
sentences in `novel_sentences.md` were written from scratch and appear in neither
split. 16 of 20 shared ≥50% of the reference words; none returned empty output.

Good:

| English | Model |
|---|---|
| Where is the station? | Waar is het station ? |
| We were very happy yesterday. | Wij waren gisteren al zeer gelukkig . |
| I have two brothers and one sister. | Ik heb twee broeders en een zuster . |
| I have never seen such a large dog. | Ik heb nooit zulk een hond gezien . |
| The man who came yesterday is my friend. | De man , die gisteren is , is reeds mijn vriend ! |
| He is taller than his brother. | Hij is veel grooter dan zijn broer . |

Note it produces *19th-century literary* Dutch (`broeders`, `reeds`, `zulk`) —
it learned the register of `opus_books`, not modern Dutch.

Bad, and diagnostic:

| English | Model | What broke |
|---|---|---|
| I drink coffee every morning. | Ik drinken iedere . | `drink` vs `drinken` — no subwords |
| She gave me a beautiful gift. | Zij gaf mij een lief meisje . | `gift` → `meisje` ("girl") |
| This is a very interesting book. | Dit is een mooie boek . | wrong adjective, agreement error |
| The colour of the sky is blue. | De kleur van het uitspansel . | dropped the predicate |

---

## 6. What to change next

In priority order:

1. **Switch to a BPE tokenizer.** This attacks the four tokenizer flaws at once
   and is the single highest-value change. `graphs/bpe/README.md` has the plan
   and the baseline to beat (BLEU-4 24.21).
2. **Log validation every epoch and keep every checkpoint.** The current run lost
   54% of its loss data and 17 of 20 BLEU values, which is why the peak cannot be
   located more precisely than "17, 18 or 19".
   **Partly done:** validation now runs and scores all 200 sentences every epoch
   with the fixed `corpus_bleu`, and `train.py` now appends one row per epoch to
   `training_metrics.csv`, which survives a disconnect. Still missing: a
   per-epoch checkpoint archive. Each checkpoint is 838 MB because it carries
   Adam's two moments, so only 3 of 20 fit on disk — a weights-only checkpoint
   would be 294 MB and all 20 would fit.
3. **Stop at the peak, or early-stop on validation BLEU.** Two of twenty epochs
   were pure overfitting.
4. **Weight-tie the projection layer to the target embedding** — saves 10.07M
   parameters (13.7% of the model) and often improves BLEU slightly.
5. **Add embedding dropout** (currently 0.1 only inside the blocks and positional
   encoding) and lower the LR or add a schedule near epoch 15.
6. **Add a KV cache.** Decoding re-runs the whole prefix every step, so it is
   O(n²) in generated length.
7. **Report a BPE-vs-WordLevel delta at matched training budget** (same 20 epochs,
   same seed, same `seq_len`), comparing at epoch 17, not epoch 19.

---

## 7. Reproduce

```bash
cd pytorch-transformer
python tools/evaluate.py --checkpoint tmodel_17.pt --samples 200   # BLEU/CER/WER
python tools/compare_checkpoints.py --checkpoints \
    model/tmodel_17.pt drive/tmodel_18.pt drive/tmodel_19.pt --samples 60
python tools/novel_sentences.py --checkpoint 17                    # never-seen sentences
python translate.py --checkpoint 17 "The cat sleeps on the table."
python tools/make_graphs.py                                        # training curves
python tools/make_attention_graphs.py                              # attention heatmaps
python tools/write_model_card.py                                   # model/model_card.json
```

The `tools/` scripts import the implementation from the parent directory and
resolve every output path absolutely, so they also run from any working
directory.
