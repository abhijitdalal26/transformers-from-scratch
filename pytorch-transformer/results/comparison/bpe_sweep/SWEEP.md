# BPE checkpoint sweep - every checkpoint on disk, one protocol

- **Decoding:** greedy | **Split:** seeded (seed 42) 90/10 split of `Helsinki-NLP/opus_books` (en-nl) | **Sentences:** 200 held-out
- **Case-blind** = prediction and reference lowercased before scoring (the BPE tokenizer lowercases its output, the WordLevel one does not)
- Produced by `tools/bpe_sweep.py`

## Scores

| Metric | bpe_ep9 | bpe_ep10 | bpe_ep12 | bpe_ep16 | bpe_ep17 | bpe_ep18 | bpe_ep19 | wordlevel_ep17 |
|---|---|---|---|---|---|---|---|---|
| variant | bpe | bpe | bpe | bpe | bpe | bpe | bpe | wordlevel |
| epoch | 9 | 10 | 12 | 16 | 17 | 18 | 19 | 17 |
| global_step | 43490 | 47839 | 56537 | 73933 | 78282 | 82631 | 86980 | 78282 |
| params (M) | 56.4 | 56.4 | 56.4 | 56.4 | 56.4 | 56.4 | 56.4 | 73.14 |
| vocab src / tgt | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 | 17396 / 19637 |
| **BLEU-4 (cased)** | 8.60 | 8.32 | 6.97 | 7.99 | 8.01 | 6.94 | 7.72 | 24.21 |
| **BLEU-4 (case-blind)** | 11.55 | 11.01 | 9.93 | 10.60 | 10.94 | 9.70 | 10.19 | 24.43 |
| BLEU-1 prec (cased) | 0.3942 | 0.3824 | 0.3651 | 0.3796 | 0.3707 | 0.3751 | 0.3773 | 0.5372 |
| BLEU-4 prec (cased) | 0.0235 | 0.0224 | 0.0148 | 0.0206 | 0.0202 | 0.0147 | 0.0195 | 0.1294 |
| brevity penalty | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| CER (cased) | 0.6680 | 0.6990 | 0.7244 | 0.6891 | 0.7135 | 0.6938 | 0.6863 | 0.5686 |
| WER (cased) | 0.9434 | 0.9955 | 1.0140 | 0.9693 | 1.0069 | 0.9753 | 0.9703 | 0.9196 |
| CER (case-blind) | 0.6586 | 0.6901 | 0.7144 | 0.6804 | 0.7048 | 0.6850 | 0.6774 | 0.5671 |
| WER (case-blind) | 0.9106 | 0.9650 | 0.9798 | 0.9408 | 0.9757 | 0.9436 | 0.9401 | 0.9175 |
| exact match (cased) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mean gen len (tokens) | 26.46 | 28.59 | 30.08 | 27.88 | 29.45 | 27.75 | 28.59 | 24.77 |
| mean gen len (words) | 20.53 | 22.38 | 23.2 | 21.14 | 22.45 | 21.25 | 21.55 | 24.2 |
| mean ref len (words) | 21.02 | 21.02 | 21.02 | 21.02 | 21.02 | 21.02 | 21.02 | 21.02 |
| src [UNK] rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0127 |
| ref [UNK] rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0160 |
| looped sentences (4-gram) | 18 | 20 | 18 | 11 | 20 | 9 | 8 | 9 |
| empty predictions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| never-seen hits >=50% ref | 16 | 16 | 15 | 13 | 16 | 15 | 16 | 16 |
| decode seconds | 2215.5 | 2602.6 | 2806.1 | 660.9 | 677.8 | 617.0 | 643.1 | 662.2 |

## Verdict

- **Best cased BLEU-4 (all variants):** `tmodel_17.pt` (epoch 17) - 24.21
- **Best case-blind BLEU-4 (all variants):** `tmodel_17.pt` (epoch 17) - 24.43
- **Best BPE checkpoint:** `tmodel_09.pt` (epoch 9) - 8.60 cased / 11.55 case-blind

## Head-to-head per sentence (case-blind BLEU)

Computed over the 3 checkpoints that stored their decoded sentences.

| checkpoint | sentences won |
|---|---|
| `bpe_ep9` | 31 |
| `bpe_ep10` | 31 |
| `bpe_ep12` | 20 |
| tied | 118 |

## Never-seen sentences (model output)

| English | Reference | `bpe_ep9` | `bpe_ep10` | `bpe_ep12` | `bpe_ep16` | `bpe_ep17` | `bpe_ep18` | `bpe_ep19` | `wordlevel_ep17` |
|---|---|---|---|---|---|---|---|---|---|
| The cat sleeps on the table. | De kat slaapt op de tafel. | de kat was op de tafel. | de kat slaapt op de tafel. | de schroeven slaapt op de tafel. | de roede van tafel staat op de tafel. | de schrei van de tafel op de tafel. | de kat ging zitten op de tafel. | de roede van den tafel op. | De kat , die aan de tafel stond . |
| I drink coffee every morning. | Ik drink elke morgen koffie. | ik baadde den 11den dag. | ik heb den morgen al wat hij deed. | ik dronk den goeden morgen. | ik ontken den morgen om ieder woord. | ik maakte een gereed om alles te schieten. | ik ga morgen den goeden morgen aan. | ik dronk den des morgens ochtends. | Ik drinken iedere . |
| She does not like fish. | Zij houdt niet van vis. | zij zijn niet zoo visschen. | zij zijn niet zoo visschen. | zij ziet er niet zooals visschen. | zij ziet er niet zooals visschen. | zij vischt het niet zooals wij weten. | zij ziet er niet zooals visch. | zij ziet er niet zooals visschen. | Zij doet er niet toe . |
| We were very happy yesterday. | Wij waren gisteren zeer gelukkig. | het was gisteren gelukkig. | wij waren gelukkig. | wij waren gisteren gisteravond zeer gelukkig. | het was nu gisteren al gelukkig. | het was gisteren al gelukkig, wij waren gisteren zeer gelukkig. | wij waren gisteren gelukkig bepaald. | het was nu gisteren al gelukkig. | Wij waren gisteren al zeer gelukkig . |
| The book is on the shelf. | Het boek ligt op de plank. | het boek is op de borst gekomen. | het boek is op het boek gekomen. | de boek staat op het boek. | de boek is goed. | het boek is op haar beurt. | de boek is opgestaan. | de boek is op de vonnis. | Het boek ligt op den heuvel . |
| He is taller than his brother. | Hij is langer dan zijn broer. | hij is veel waard dan zijn broeder. | hij is veel waard dan zijn broer. | hij is veel grooter dan zijn broeder. | hij is grooter dan zijn broeder. | hij is veel grooter dan zijn broeder. | hij is veel grooter dan zijn broeder. | hij is grooter dan zijn broeder. | Hij is veel grooter dan zijn broer . |
| What are you doing? | Wat doe je? | wat doet gij toch? | wat doet gij? | wat doet gij? | wat doet gij toch? | wat heb je nou wel noodig? | wat heb je dan gedaan? | wat doe je dat? | Wat doet u dat ? |
| Where is the station? | Waar is het station? | waar is het station? | waar is het station? | waar is het station? | waar is het station? | waar is het station? | waar is het station? | waar is het station? | Waar is het station ? |
| I have two brothers and one sister. | Ik heb twee broers en een zuster. | ik heb twee broer en een zuster. | ik heb twee broer en een zuster ook. | ik heb twee broeders en een zuster ook. | ik heb twee broer en een zuster. | ik heb twee broeders en een zuster logeeren. | ik heb twee broederss en een zuster bij elkander gehad. | ik heb twee broederss, en een zuster. | Ik heb twee broeders en een zuster . |
| The weather is cold today. | Het weer is koud vandaag. | het weer is heden koud. | het weer is heden koud. | het weer is heden weer koud. | het weer is heden koud. | het weer is heden koud. | het weer is heden koud. | het weer is heden koud. | Het is vandaag koud . |
| They will come to the party. | Zij zullen naar het feest komen. | zij zullen den anderen kant wel eens komen. | zij zullen den anderen kant komen. | zij zullen het gezelschap bereiken. | ze zullen den anderen hier leeren, naar ik denk. | ze zullen wel hier te komen. | ze zullen den anderen kantoor leeren. | zij zullen den anderen dood tegemoet komen. | Zij zullen het aan iedereen komen . |
| This is a very interesting book. | Dit is een zeer interessant boek. | dat is een zeer belangwekkend boek. | dit is een zeer belangwekkend boek. | dit is een zeer belangwekkend boek. | dit is een zeer belangwekkend boek. | dit is een zeer belangwekkend werk. | dit is een zeer belangwekkend man. | dit is het een zeer belangwekkend boek. | Dit is een mooie boek . |
| The child is playing in the garden. | Het kind speelt in de tuin. | het kind is in den tuin. | het kind is in den tuin gekomen. | het kind is in den tuin. | het kind is in den tuin. | het kind zit in den tuin. | het kind is in den tuin. | de kind gaat naar den tuin. | Het kind is in den tuin . |
| I do not understand this word. | Ik begrijp dit woord niet. | ik begrijp niet, dat dit woord er sprake was. | ik begrijp niet. | ik begrijp niet, maar ik begrijp het niet. | ik begrijp dit woord niet. | ik begrijp niet of ik dit woord nog wel begreep. | ik begrijp dit nog niet. | ik begrijp niet dat dit woord het woord is. | Ik begrijp het niet . |
| The man who came yesterday is my friend. | De man die gisteren kwam is mijn vriend. | de man die gisteren is aangekomen. | die man is gisteren aangekomen. | de man, die gisteren is geweest, is mijn vriend! | de man, die gisteren is, is mijn vriend. | de man, die gisteren is geweest, is mijn vriend! | de man die gisteren gisteren is mijn vriend, is mijn waarde vriend! | de man, die gisteren mijn vriend is, is mijn vriend. | De man , die gisteren is , is reeds mijn vriend ! |
| Please open the window. | Open alsjeblieft het raam. | en wat het raam betreft. | als het raam openging. | doe als het venster. | en als het raam opene raam open. | betuigerlijk het raam open. | och als het raam openst. | en het raam open. | Het raam is goed . |
| The colour of the sky is blue. | De kleur van de hemel is blauw. | de kleur der hemel is een blauw fluit. | de kleur van het hemel is een blauw flink. | de vacanto van den hemel is half uitgespreid. | de blo van het hemel is een blauwe vak. | de ruhmkorff is een blauw flater. | de vacantie is blauw blauwe vak. | de vonnis der von is blauw naakt. | De kleur van het uitspansel . |
| Water boils at one hundred degrees. | Water kookt bij honderd graden. | water uit honderd graden. | water uit honderd graad. | water, honderd mijlen in honderd graad. | water van honderd graad. | water der honderd graad! | waterken op honderd°. | water van honderd graad. | Water uit honderd graad . |
| She gave me a beautiful gift. | Zij gaf mij een prachtig geschenk. | zij gaf mij een lief meisje aan te nemen. | zij gaf mij een lief, die een liefhad. | zij gaf mij een fraai briefje te kennen. | zij gaf mij een schoone aanstoot te gelooven. | zij gaf mij een mooie aanstakigheid. | zij gaf mij een fraai briefje. | zij gaf mij een schoon geschenk aan. | Zij gaf mij een lief meisje . |
| I have never seen such a large dog. | Ik heb nog nooit zo'n grote hond gezien. | ik heb nooit zoo'n hond nooit gezien. | ik heb nooit zoo'n hond gezien. | ik heb nooit zoo'n hond gezien. | ik heb nooit zoo'n grooten hond nooit gezien. | ik heb nog nooit zoo'n hond nooit gezien. | ik heb nog nooit zulk een hond gezien. | ik heb nooit zoo'n hond nooit gezien. | Ik heb nooit zulk een hond gezien . |
