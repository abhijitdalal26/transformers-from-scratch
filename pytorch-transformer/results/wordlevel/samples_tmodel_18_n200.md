# Greedy translations - tmodel_18_n200

- **Checkpoint:** `tmodel_18` (epoch 18, global_step 82631)
- **Decoding:** greedy, no beam search | **device:** cpu
- **Scored on:** 200 held-out sentences (seed 42)
- **BLEU-4:** 20.84 | **CER:** 0.580 | **WER:** 0.921 | **BP:** 0.954

Samples are sorted by closest length to the reference (easiest cases first).

### 1

- **SOURCE:** Conseil had said the true word.
- **TARGET:** Koenraad had het juiste woord gekozen.
- **PREDICTED:** Koenraad had het juiste woord gezegd .

### 2

- **SOURCE:** Athos and d�Artagnan went out, while Planchet was sent to inquire if there were not three horses for sale in the neighborhood.
- **TARGET:** Athos en d�Artagnan gingen uit, terwijl Planchet ging vernemen, of er niet een drietal paarden in den omtrek te koop waren.
- **PREDICTED:** Athos en d � Artagnan gingen uit , terwijl Planchet was , alsof er geen drie paarden in de nabijheid van den omtrek waren .

### 3

- **SOURCE:** "Whom do you think that is to?" he asked, as we resumed our journey.
- **TARGET:** �Aan wien denkt gij dat het gericht was?" vroeg hij, toen wij verder reden.
- **PREDICTED:** " Wie denkt u dat het is ?" vroeg hij , toen wij ons vertrek weder verder .
