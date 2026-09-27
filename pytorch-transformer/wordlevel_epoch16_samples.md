# Sample translations - WordLevel tokenizer

- **Tokenizer:** WordLevel (en 17396 / nl 19637 words, `min_frequency=2`)
- **Checkpoint:** `tmodel_16.pt`
- **Epoch:** 16  |  **global_step:** 73933
- **Decoding:** greedy, no beam search
- **Source:** Helsinki-NLP/opus_books (en-nl)

Baseline for later comparison against a BPE-tokenizer run.

---

## Model facts and numbers

### Architecture / hyperparameters

| Setting | Value |
|---|---|
| `d_model` | 512 |
| layers (N) | 6 encoder + 6 decoder |
| attention heads (h) | 8 |
| `d_ff` | 2048 |
| dropout | 0.1 |
| `seq_len` | 350 |
| `batch_size` | 8 |
| learning rate | 1e-4 (constant, no schedule) |
| label smoothing | 0.1 |
| dataset | Helsinki-NLP/opus_books (en-nl) |
| total pairs | 38,652 |
| split | 90 / 10, **unseeded** at the time of this run |
| train pairs | 34,786 |
| steps per epoch | 4,349 |
| epochs completed | 17 (0-16) |
| final global_step | 73,933 |
| device | Colab GPU |

### Vocabulary (this is the problem area)

| | source (en) | target (nl) |
|---|---|---|
| vocabulary size | 17,396 | 19,637 |
| entries actually used | 9,671 (56%) | 11,764 (60%) |
| **unused / dead entries** | **~7,725 (44%)** | **~7,873 (40%)** |
| words seen exactly once | 2,682 (28% of used) | 3,402 (29% of used) |
| `[UNK]` rate | 0.60% of tokens | 1.34% of tokens |
| avg tokens per sentence | 38.6 | 36.0 |

`min_frequency=2` was used, so any word appearing only once was dropped and became `[UNK]`.

`ln(vocab_tgt) = 9.885` — this is why training loss starts at ~9.9.

### Tokenizer flaws found

- **No lowercasing.** `tokenizer.normalizer` is `None`, so `zij` and `Zij` are two
  separate vocabulary entries. **981 words are stored in both cases.**
- **No subwords.** `pre_tokenizer` is `Whitespace()`, so one word = one token,
  forever. `lopende`, `lopend`, `gelopen` share no representation.
- **Corpus noise in the vocabulary.** 9 entries are OCR/dialogue artifacts such
  as `--„`, `„‚`, `,--„`.

### Parameter count (measured, not estimated)

| Component | Parameters | Share |
|---|---|---|
| source embedding | 8,906,752 (8.91M) | 12.2% |
| target embedding | 10,054,144 (10.05M) | 13.7% |
| encoder (6 blocks) | 18,903,040 (18.90M) | 25.9% |
| decoder (6 blocks) | 25,200,640 (25.20M) | 34.5% |
| projection | 10,073,781 (10.07M) | 13.8% |
| **TOTAL** | **73,138,357 (73.14M)** | 100% |

- **Embeddings: 18,960,896 (18.96M) = 25.9% of the model**
- Non-embedding: 54,177,461 (54.18M) = 74.1%
- The projection layer is a separate 19,637 x 512 matrix and is *not* weight-tied
  to the target embedding, so the target vocabulary is paid for twice
  (10.05M + 10.07M = 20.1M, or 27.5% of the model).
- Checkpoint on disk: **838.6 MB** each (73.14M params x 4 bytes x 3, for weights
  + Adam m + Adam v = 837 MB).
- 1,024 embedding parameters per token id (512 x 2 vocabularies).

**The core problem in one line: 18.96M embedding parameters (26% of the model)
learned from only 34,786 training sentences, where 40% of the rows are never
touched and another 29% are touched once.**

## Scores achieved

Measured with a verified BLEU-4 (the `torchmetrics` BLEU in this repo returns
0.0 even for identical text, so it is unusable).

**In-domain** — 150 random opus_books pairs, seed 1234, greedy decoding:

| Checkpoint | Epoch | BLEU-4 | p1 | p2 | p3 | p4 | BP | CER | WER |
|---|---|---|---|---|---|---|---|---|---|
| `tmodel_15.pt` | 15 | **30.07** | 59.7 | 35.7 | 25.9 | 19.7 | 0.931 | 0.519 | 0.858 |
| `tmodel_16.pt` | 16 | **31.21** | 59.6 | 36.4 | 26.3 | 20.3 | 0.951 | 0.521 | 0.871 |

Still improving: **+1.14 BLEU** in one epoch, and 124/150 outputs changed.

**Out-of-domain** — the 5 hand-written sentences in the second half of this file:

| Metric | Value |
|---|---|
| BLEU-4 | 19.90 |
| p1 / p2 / p3 / p4 | 70.7 / 37.7 / 16.7 / 4.7 |
| BP | 0.933 |
| CER | 0.410 |
| WER | 0.526 |
| word recall | 66.1% (41/62) |
| sentences fully correct | 0 / 5 |

**Train loss:** 9.898 (epoch 0 start) -> 3.051 (epoch 9 end, last logged point).

### Caveat on these numbers

The in-domain set is drawn from the same corpus the model trained on, and the
original 90/10 split used **no seed**, so the true held-out split cannot be
reconstructed. Treat 31.21 as an optimistic in-domain figure, not a clean
held-out score. The out-of-domain number (19.90) is the honest generalisation
estimate.

## Why this model plateaus, and why BPE is the fix

### The diagnosis

The model **is** genuinely learning, and it is not memorising:

- It still improves epoch over epoch (+1.14 BLEU at epoch 16).
- It scores 19.90 BLEU on sentences that appear nowhere in the training corpus.
  Pure memorisation would score 0 there.
- 4-gram precision of 20.3% is real structural overlap, not isolated word hits.

But it is **under-trained for its own vocabulary**, and it has learned the
*style* of one 19th-century literary translation rather than translation itself.
It writes `grijsaard` and `den tuin` where a modern sentence wants `oude man`
and `de tuin`. In-domain 31.21 vs out-of-domain 19.90 is that gap.

### Root cause: WordLevel on a small corpus

A 37k-word vocabulary on 34,786 training sentences is the ceiling. Each of
those words is an all-or-nothing slot: either the exact string is in the list,
or the model is told "a word is here" and given nothing.

1. **~40% of embedding rows are dead.** ~7,900 target rows never appear in the
   data, so they never receive a gradient and stay at their random init.
2. **29% of remaining words appear once.** One occurrence, one gradient, no
   useful learning.
3. **1.34% of target tokens are `[UNK]`.** Real information is deleted outright.
4. **Inflections are unrelated tokens.** `huis` / `huizen` / `huisje` share
   nothing, so the model learns each from scratch.

### What BPE changes

BPE learns character chunks instead of whole words, repeatedly merging the most
frequent adjacent pair until the vocabulary budget is met.

Same sentence, both tokenizers:

```
"Het regende hard toen zij het huis binnenkwam."
  WordLevel -> Het regende hard toen zij het huis binnenkwam .     9 tokens
  BPE       -> Het regende hard toen zij het huis binnen kwam .  10 tokens

"Zij stak den brief in zijn zak en ging voort lopende."
  WordLevel -> Zij stak den brief in zijn zak en ging voort [UNK] .   <-- word lost
  BPE       -> Zij stak den brief in zijn zak en ging voort l opende .
```

Measured effect of moving to BPE with an 8,000-token vocabulary:

| | WordLevel (current) | BPE 8000 (proposed) |
|---|---|---|
| `[UNK]` rate | 1.34% | **0.00%** |
| dead embedding rows | ~7,900 | ~0 (every piece is frequent) |
| embedding parameters | 18.96M (25.9%) | **8.19M (11.2%)** |
| parameters saved | — | **10,768,896 (57% fewer)** |

BPE can never produce `[UNK]` because it falls back all the way to characters,
and rare words decompose into pieces the model *has* learned. That is also why
it is a big win here specifically: with 34,786 sentences you cannot learn 37k
embeddings, but you can learn 16k.

### Honest trade-off

BPE makes sentences slightly longer and each token less meaningful, so early
training is slower and it wants more data. On a small corpus the gain is smaller
than on a large one — typically **+5 to +10 BLEU** in this situation, which would
put a from-scratch BPE run in the high 30s versus the current 31.

### Also worth doing

Add `tokenizer.normalizer = Lowercase()`. This alone recovers ~981 wasted
vocabulary slots and is free.

## Bugs fixed in this repo (commit `3ed8ca9`)

| Bug | Effect |
|---|---|
| `torchmetrics` 2.21 `BLEUScore` returns 0.0 for identical text | every epoch logged BLEU 0.00 |
| `run_validation` scored only 2 sentences from a shuffled loader | CER/WER were pure noise |
| `ignore_index` used the source `[PAD]`, not the target's | only worked by luck |
| val DataLoader shuffled | metrics not comparable across epochs |
| `random_split` unseeded | held-out scores unreproducible |
| `greedy_decode` could emit `[PAD]` and returned the leading `[SOS]` | polluted outputs and scores |

A Noam learning-rate schedule was added but is **opt-in and off by default**.
Enabling it on a resume from step 73,933 would set the LR to **1.63e-08**
instead of 1e-4 — 6,000x smaller, which would freeze training and look like the
model had stopped learning. Noam is only safe when training from scratch.

---

## In-domain (real opus_books sentences)

### 1

**EN** &nbsp; "By my faith, I ask no better," cried Bonacieux; "I shall not be sorry to see the face of an acquaintance."

**TARGET** &nbsp; --„Gelukkig is de heer d’Artagnan in onze handen en men zal hem in uw tegenwoordigheid brengen.”

**PREDICTED** &nbsp; --„ Op mijn woord , ik vraag niets ,” zeide Bonacieux , „ ik zal het niet eens zien , dat mij het gezicht zal zien .”

### 2

**EN** &nbsp; "Begone, then, find them instantly, and let all be done tonight!

**TARGET** &nbsp; --„Ga hen dus oogenblikkelijk zoeken, en dat nog dezen nacht alles worde uitgevoerd.

**PREDICTED** &nbsp; --„ Ga hen dan oogenblikkelijk zoeken , laat alles van nacht schijnen .

### 3

**EN** &nbsp; Didn't I tell you!" cried the poor little man, throwing out his hands, and looking from one to the other of us.

**TARGET** &nbsp; Heb ik het u niet gezegd!" schreide het arme manktetje, zijne handen uitstrekkende en elk onzer beurtelings in het gelaat ziende.

**PREDICTED** &nbsp; Heb ik het u niet gezegd !" riep de arme man , zijne handen uitstrekkende en elk onzer beurtelings in het gelaat ziende .

### 4

**EN** &nbsp; Poor girl, she did not know how fast she was nearing trouble herself.

**TARGET** &nbsp; Arm kind! zij wist niet, hoezeer zij op weg was zich een wereld van verdriet te bezorgen.

**PREDICTED** &nbsp; Arm kind ! zij wist niet , hoe zij zich zou wagen om haar te winnen .

### 5

**EN** &nbsp; And I believe Marianne will be the most happy with him of the two."

**TARGET** &nbsp; En ik geloof dat Marianne nog het gelukkigst met hem zal zijn."

**PREDICTED** &nbsp; En ik geloof dat Marianne ' s met hem zal zijn ."

### 6

**EN** &nbsp; Much rain also in the days, though with some intervals of fair weather; but it seems this was the rainy season.

**TARGET** &nbsp; Veel regen in deze dagen, doch tusschenbeide mooi weder; het schijnt, dat dit het regenachtige jaargetijde is.

**PREDICTED** &nbsp; Veel regen in de mode , hoewel met eenige ; maar het schijnt dat dit het regenachtige jaargetijde is .

### 7

**EN** &nbsp; M. de Treville required this intelligence to be repeated to him twice, and each time his companions saw his brow become darker.

**TARGET** &nbsp; De Tréville liet zich dat nieuws tot tweemaal toe herhalen, waarbij telkens zijn krijgslieden zijn gelaat zagen verduisteren.

**PREDICTED** &nbsp; De heer de Tréville bracht deze heerlijke tijding mede en beide malen zagen elkander aan zijn voorhoofd .

### 8

**EN** &nbsp; ABOUT noon the next day the boys arrived at the dead tree; they had come for their tools.

**TARGET** &nbsp; Den volgenden dag, tegen twaalf uren, stonden de knapen bij den dooden boom om hun gereedschap te halen.

**PREDICTED** &nbsp; Den volgenden dag , den anderen morgen , kwamen de knapen bij den dooden boom op .

### 9

**EN** &nbsp; Mrs. John Dashwood said nothing; but her husband civilly hoped that she would not be settled far from Norland.

**TARGET** &nbsp; Mevrouw John Dashwood zei niets; doch haar echtgenoot gaf beleefd zijn hoop te kennen, dat zij niet ver van Norland wonen zou.

**PREDICTED** &nbsp; Mevrouw John Dashwood zei niets ; maar haar echtgenoot wist niet , dat zij niet ver van Norland wonen zou .

### 10

**EN** &nbsp; No mere ferule will answer for this offence.

**TARGET** &nbsp; Dat kan met de roede alleen niet afgedaan worden.

**PREDICTED** &nbsp; Geen enkel vooruitzicht zal voor deze dwaze vraag geen blijk geven .

### 11

**EN** &nbsp; At the end of an instant’s observation he heard the noise of a vehicle, and saw Milady’s carriage stop opposite to him.

**TARGET** &nbsp; Na een oogenblik achter de haag een en ander te hebben gadegeslagen, hoorde hij het gerucht van een naderend rijtuig, en hij zag voor zich de koets van milady stil houden.

**PREDICTED** &nbsp; Na verloop van een oogenblik hoorde hij het gerucht van een , en zag milady het rijtuig voor hem staan .

### 12

**EN** &nbsp; Poll, as if he had been my favourite, was the only person permitted to talk to me.

**TARGET** &nbsp; Ik at, als een koning, geheel alleen, omringd door al mijne dienaren!

**PREDICTED** &nbsp; Mijn papegaai , als hij mijn dank had , was de eenige persoon , die mij naar willekeur had kunnen voeren .

---

## Out-of-domain (hand-written, not in the training corpus)

### 1

**EN** &nbsp; The old man opened the wooden door and walked into the garden.

**TARGET** &nbsp; De oude man opende de houten deur en liep de tuin in.

**PREDICTED** &nbsp; De grijsaard opende de deur en stapte in den tuin van den tuin .

### 2

**EN** &nbsp; She had never seen such a beautiful sunset above the mountains.

**TARGET** &nbsp; Zij had nooit zoo'n mooie zonsondergang boven de bergen gezien.

**PREDICTED** &nbsp; Zij had nooit zulk een prachtig gezien .

### 3

**EN** &nbsp; He put the letter into his pocket and continued walking.

**TARGET** &nbsp; Hij stak den brief in zijn zak en ging voort lopende.

**PREDICTED** &nbsp; Hij legde den brief op zijn zak en ging verder voort .

### 4

**EN** &nbsp; My name is Anna and I live in a small village near the sea.

**TARGET** &nbsp; Mijn naam is Anna en ik woon in een klein dorp aan zee.

**PREDICTED** &nbsp; Mijn naam is en ik leven in een klein gehucht .

### 5

**EN** &nbsp; The children were playing in the garden when it began to rain.

**TARGET** &nbsp; De kinderen speelden in den tuin toen het begon te regenen.

**PREDICTED** &nbsp; De kinderen werden in den tuin , toen het begon te worden .
