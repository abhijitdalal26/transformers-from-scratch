# Side-by-side inference - tmodel_17 vs tmodel_19

The same 20 sentences translated by each checkpoint with greedy decoding.

These are the never-seen sentences from `novel_sentences.py`: written by hand,
appearing in neither the training nor the validation split. Per-sentence BLEU
is measured against the hand-written Dutch reference. The model was trained on
19th-century literary prose and answers in that register, so the absolute BLEU
is low against modern references - the point here is that every checkpoint is
judged on the identical yardstick.

| English | Reference | `tmodel_17` | `tmodel_19` |
|---|---|---|---|
| The cat sleeps on the table. | De kat slaapt op de tafel. | De kat , die aan de tafel stond . | De kat , tegen de tafel . |
| I drink coffee every morning. | Ik drink elke morgen koffie. | Ik drinken iedere . | Ik drinken iedere per morgen . |
| She does not like fish. | Zij houdt niet van vis. | Zij doet er niet toe . | Zij doet er niet toe als visschen . |
| We were very happy yesterday. | Wij waren gisteren zeer gelukkig. | Wij waren gisteren al zeer gelukkig . | Wij waren gisteren al zeer gelukkig . |
| The book is on the shelf. | Het boek ligt op de plank. | Het boek ligt op den heuvel . | Het boek staat op den lei : |
| He is taller than his brother. | Hij is langer dan zijn broer. | Hij is veel grooter dan zijn broer . | Hij is dan zijn broeder . |
| What are you doing? | Wat doe je? | Wat doet u dat ? | Wat heb je nou te doen ? |
| Where is the station? | Waar is het station? | Waar is het station ? | Waar is het station ? |
| I have two brothers and one sister. | Ik heb twee broers en een zuster. | Ik heb twee broeders en een zuster . | Ik heb twee broeders en een zuster . |
| The weather is cold today. | Het weer is koud vandaag. | Het is vandaag koud . | Het weer is heden koud . |
| They will come to the party. | Zij zullen naar het feest komen. | Zij zullen het aan iedereen komen . | Ze zullen het gezelschap worden . |
| This is a very interesting book. | Dit is een zeer interessant boek. | Dit is een mooie boek . | Dit is een bijzonder boek . |
| The child is playing in the garden. | Het kind speelt in de tuin. | Het kind is in den tuin . | Het kind staat in den tuin . |
| I do not understand this word. | Ik begrijp dit woord niet. | Ik begrijp het niet . | Ik begrijp zulks niet . |
| The man who came yesterday is my friend. | De man die gisteren kwam is mijn vriend. | De man , die gisteren is , is reeds mijn vriend ! | De man die gisteren is komen is mijn vriend ! |
| Please open the window. | Open alsjeblieft het raam. | Het raam is goed . | Toe , het raam is goed . |
| The colour of the sky is blue. | De kleur van de hemel is blauw. | De kleur van het uitspansel . | De kleur van het uitspansel . |
| Water boils at one hundred degrees. | Water kookt bij honderd graden. | Water uit honderd graad . | Water uit honderd graad ! |
| She gave me a beautiful gift. | Zij gaf mij een prachtig geschenk. | Zij gaf mij een lief meisje . | Zij heeft mij een fraaie genegenheid voor mij gegeven . |
| I have never seen such a large dog. | Ik heb nog nooit zo'n grote hond gezien. | Ik heb nooit zulk een hond gezien . | Ik heb zulk een hond gezien . |

## Per-sentence BLEU against the reference

| English | `tmodel_17` | `tmodel_19` | better |
|---|---|---|---|
| The cat sleeps on the table. | 0.0 | 0.0 | `tmodel_17` |
| I drink coffee every morning. | 0.0 | 0.0 | `tmodel_17` |
| She does not like fish. | 0.0 | 0.0 | `tmodel_17` |
| We were very happy yesterday. | 0.0 | 0.0 | `tmodel_17` |
| The book is on the shelf. | 43.5 | 0.0 | `tmodel_17` |
| He is taller than his brother. | 41.1 | 0.0 | `tmodel_17` |
| What are you doing? | 0.0 | 0.0 | `tmodel_17` |
| Where is the station? | 100.0 | 100.0 | `tmodel_17` |
| I have two brothers and one sister. | 50.0 | 50.0 | `tmodel_17` |
| The weather is cold today. | 0.0 | 0.0 | `tmodel_17` |
| They will come to the party. | 0.0 | 0.0 | `tmodel_17` |
| This is a very interesting book. | 0.0 | 0.0 | `tmodel_17` |
| The child is playing in the garden. | 0.0 | 0.0 | `tmodel_17` |
| I do not understand this word. | 0.0 | 0.0 | `tmodel_17` |
| The man who came yesterday is my friend. | 0.0 | 38.0 | `tmodel_19` |
| Please open the window. | 0.0 | 0.0 | `tmodel_17` |
| The colour of the sky is blue. | 0.0 | 0.0 | `tmodel_17` |
| Water boils at one hundred degrees. | 0.0 | 0.0 | `tmodel_17` |
| She gave me a beautiful gift. | 43.5 | 0.0 | `tmodel_17` |
| I have never seen such a large dog. | 0.0 | 0.0 | `tmodel_17` |

## Verdict

| | mean per-sentence BLEU | sentences won |
|---|---|---|
| `tmodel_17` | 13.90 | 19 / 20 |
| `tmodel_19` | 9.40 | 1 / 20 |

**tmodel_17** wins 19 of 20 sentences.
