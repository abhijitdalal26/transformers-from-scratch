# Variant comparison - wordlevel_tmodel_17 | bpe_tmodel_16 | bpe_tmodel_17 | bpe_tmodel_18 | bpe_tmodel_19

- **Protocol:** greedy decoding, no beam search, 200 identical held-out sentences
- **Split:** seeded (seed 42) 90/10 split of `Helsinki-NLP/opus_books` (en-nl)
- **Case-blind** = prediction and reference lowercased before scoring (the BPE tokenizer lowercases its output, the WordLevel one does not)

## Scores

| Metric | wordlevel_tmodel_17 | bpe_tmodel_16 | bpe_tmodel_17 | bpe_tmodel_18 | bpe_tmodel_19 |
|---|---|---|---|---|---|
| epoch | 17 | 16 | 17 | 18 | 19 |
| global_step | 78282 | 73933 | 78282 | 82631 | 86980 |
| params (M) | 73.14 | 56.4 | 56.4 | 56.4 | 56.4 |
| vocab src / tgt | 17396 / 19637 | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 | 8000 / 8000 |
| BLEU-4 (cased) | 24.21 | 7.99 | 8.01 | 6.94 | 7.72 |
| BLEU-4 (case-blind) | 24.43 | 10.60 | 10.94 | 9.70 | 10.19 |
| BLEU-1 prec | 0.5421 | 0.4144 | 0.4044 | 0.4107 | 0.4125 |
| BLEU-2 prec | 0.2882 | 0.1479 | 0.1467 | 0.1378 | 0.1402 |
| BLEU-3 prec | 0.1838 | 0.0636 | 0.0677 | 0.0570 | 0.0615 |
| BLEU-4 prec | 0.1305 | 0.0323 | 0.0357 | 0.0274 | 0.0303 |
| brevity penalty | 0.9873 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CER | 0.5671 | 0.6804 | 0.7048 | 0.6850 | 0.6774 |
| WER | 0.9175 | 0.9408 | 0.9757 | 0.9436 | 0.9401 |
| exact match | 0 | 4 | 4 | 3 | 4 |
| mean gen len (tokens) | 24.77 | 27.88 | 29.45 | 27.75 | 28.59 |
| mean gen len (words) | 24.2 | 21.14 | 22.45 | 21.25 | 21.55 |
| src [UNK] rate | 0.0127 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ref [UNK] rate | 0.0160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| looped sentences (4-gram) | 9 | 11 | 20 | 9 | 8 |
| empty predictions | 0 | 0 | 0 | 0 | 0 |
| never-seen hits >=50% ref | 16 | 13 | 16 | 15 | 16 |
| decode seconds | 662.2 | 660.9 | 677.8 | 617.0 | 643.1 |

## Verdict

**Best on case-blind BLEU-4: `wordlevel_tmodel_17` (epoch 17) - 24.43**

## Head-to-head per sentence (case-blind BLEU)

| checkpoint | sentences won | best on |
|---|---|---|
| `wordlevel_tmodel_17` | 89 | bpe_tmodel_17 |
| `bpe_tmodel_16` | 7 | bpe_tmodel_17 |
| `bpe_tmodel_17` | 15 | bpe_tmodel_17 |
| `bpe_tmodel_18` | 4 | bpe_tmodel_17 |
| `bpe_tmodel_19` | 5 | bpe_tmodel_17 |
| tied | 80 | |

## Never-seen sentences

| English | Reference | `wordlevel_tmodel_17` | `bpe_tmodel_16` | `bpe_tmodel_17` | `bpe_tmodel_18` | `bpe_tmodel_19` |
|---|---|---|---|---|---|---|
| The cat sleeps on the table. | De kat slaapt op de tafel. | De kat , die aan de tafel stond . | de roede van tafel staat op de tafel. | de schrei van de tafel op de tafel. | de kat ging zitten op de tafel. | de roede van den tafel op. |
| I drink coffee every morning. | Ik drink elke morgen koffie. | Ik drinken iedere . | ik ontken den morgen om ieder woord. | ik maakte een gereed om alles te schieten. | ik ga morgen den goeden morgen aan. | ik dronk den des morgens ochtends. |
| She does not like fish. | Zij houdt niet van vis. | Zij doet er niet toe . | zij ziet er niet zooals visschen. | zij vischt het niet zooals wij weten. | zij ziet er niet zooals visch. | zij ziet er niet zooals visschen. |
| We were very happy yesterday. | Wij waren gisteren zeer gelukkig. | Wij waren gisteren al zeer gelukkig . | het was nu gisteren al gelukkig. | het was gisteren al gelukkig, wij waren gisteren zeer gelukkig. | wij waren gisteren gelukkig bepaald. | het was nu gisteren al gelukkig. |
| The book is on the shelf. | Het boek ligt op de plank. | Het boek ligt op den heuvel . | de boek is goed. | het boek is op haar beurt. | de boek is opgestaan. | de boek is op de vonnis. |
| He is taller than his brother. | Hij is langer dan zijn broer. | Hij is veel grooter dan zijn broer . | hij is grooter dan zijn broeder. | hij is veel grooter dan zijn broeder. | hij is veel grooter dan zijn broeder. | hij is grooter dan zijn broeder. |
| What are you doing? | Wat doe je? | Wat doet u dat ? | wat doet gij toch? | wat heb je nou wel noodig? | wat heb je dan gedaan? | wat doe je dat? |
| Where is the station? | Waar is het station? | Waar is het station ? | waar is het station? | waar is het station? | waar is het station? | waar is het station? |
| I have two brothers and one sister. | Ik heb twee broers en een zuster. | Ik heb twee broeders en een zuster . | ik heb twee broer en een zuster. | ik heb twee broeders en een zuster logeeren. | ik heb twee broederss en een zuster bij elkander gehad. | ik heb twee broederss, en een zuster. |
| The weather is cold today. | Het weer is koud vandaag. | Het is vandaag koud . | het weer is heden koud. | het weer is heden koud. | het weer is heden koud. | het weer is heden koud. |
| They will come to the party. | Zij zullen naar het feest komen. | Zij zullen het aan iedereen komen . | ze zullen den anderen hier leeren, naar ik denk. | ze zullen wel hier te komen. | ze zullen den anderen kantoor leeren. | zij zullen den anderen dood tegemoet komen. |
| This is a very interesting book. | Dit is een zeer interessant boek. | Dit is een mooie boek . | dit is een zeer belangwekkend boek. | dit is een zeer belangwekkend werk. | dit is een zeer belangwekkend man. | dit is het een zeer belangwekkend boek. |
| The child is playing in the garden. | Het kind speelt in de tuin. | Het kind is in den tuin . | het kind is in den tuin. | het kind zit in den tuin. | het kind is in den tuin. | de kind gaat naar den tuin. |
| I do not understand this word. | Ik begrijp dit woord niet. | Ik begrijp het niet . | ik begrijp dit woord niet. | ik begrijp niet of ik dit woord nog wel begreep. | ik begrijp dit nog niet. | ik begrijp niet dat dit woord het woord is. |
| The man who came yesterday is my friend. | De man die gisteren kwam is mijn vriend. | De man , die gisteren is , is reeds mijn vriend ! | de man, die gisteren is, is mijn vriend. | de man, die gisteren is geweest, is mijn vriend! | de man die gisteren gisteren is mijn vriend, is mijn waarde vriend! | de man, die gisteren mijn vriend is, is mijn vriend. |
| Please open the window. | Open alsjeblieft het raam. | Het raam is goed . | en als het raam opene raam open. | betuigerlijk het raam open. | och als het raam openst. | en het raam open. |
| The colour of the sky is blue. | De kleur van de hemel is blauw. | De kleur van het uitspansel . | de blo van het hemel is een blauwe vak. | de ruhmkorff is een blauw flater. | de vacantie is blauw blauwe vak. | de vonnis der von is blauw naakt. |
| Water boils at one hundred degrees. | Water kookt bij honderd graden. | Water uit honderd graad . | water van honderd graad. | water der honderd graad! | waterken op honderd°. | water van honderd graad. |
| She gave me a beautiful gift. | Zij gaf mij een prachtig geschenk. | Zij gaf mij een lief meisje . | zij gaf mij een schoone aanstoot te gelooven. | zij gaf mij een mooie aanstakigheid. | zij gaf mij een fraai briefje. | zij gaf mij een schoon geschenk aan. |
| I have never seen such a large dog. | Ik heb nog nooit zo'n grote hond gezien. | Ik heb nooit zulk een hond gezien . | ik heb nooit zoo'n grooten hond nooit gezien. | ik heb nog nooit zoo'n hond nooit gezien. | ik heb nog nooit zulk een hond gezien. | ik heb nooit zoo'n hond nooit gezien. |
