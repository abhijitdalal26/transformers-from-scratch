# Never-seen sentences - `tmodel_17`

Sentences written from scratch for this test. They appear in neither the
training split nor the validation split of `Helsinki-NLP/opus_books`, so this is
a fairer read on generalisation than the validation score. Dutch references are
my own translations, not corpus gold.

| English | Dutch reference | Model output | Probes |
|---|---|---|---|
| The cat sleeps on the table. | De kat slaapt op de tafel. | De kat , die aan de tafel stond . | simple subject-verb-object |
| I drink coffee every morning. | Ik drink elke morgen koffie. | Ik drinken iedere . | frequency + habitual |
| She does not like fish. | Zij houdt niet van vis. | Zij doet er niet toe . | negation |
| We were very happy yesterday. | Wij waren gisteren zeer gelukkig. | Wij waren gisteren al zeer gelukkig . | past tense + intensifier |
| The book is on the shelf. | Het boek ligt op de plank. | Het boek ligt op den heuvel . | locative |
| He is taller than his brother. | Hij is langer dan zijn broer. | Hij is veel grooter dan zijn broer . | comparative |
| What are you doing? | Wat doe je? | Wat doet u dat ? | question |
| Where is the station? | Waar is het station? | Waar is het station ? | question + place |
| I have two brothers and one sister. | Ik heb twee broers en een zuster. | Ik heb twee broeders en een zuster . | numbers + family |
| The weather is cold today. | Het weer is koud vandaag. | Het is vandaag koud . | weather expression |
| They will come to the party. | Zij zullen naar het feest komen. | Zij zullen het aan iedereen komen . | future tense |
| This is a very interesting book. | Dit is een zeer interessant boek. | Dit is een mooie boek . | attributive adjective |
| The child is playing in the garden. | Het kind speelt in de tuin. | Het kind is in den tuin . | present continuous |
| I do not understand this word. | Ik begrijp dit woord niet. | Ik begrijp het niet . | negation + separable verb |
| The man who came yesterday is my friend. | De man die gisteren kwam is mijn vriend. | De man , die gisteren is , is reeds mijn vriend ! | relative clause |
| Please open the window. | Open alsjeblieft het raam. | Het raam is goed . | polite imperative |
| The colour of the sky is blue. | De kleur van de hemel is blauw. | De kleur van het uitspansel . | nested possessives |
| Water boils at one hundred degrees. | Water kookt bij honderd graden. | Water uit honderd graad . | numbers as words |
| She gave me a beautiful gift. | Zij gaf mij een prachtig geschenk. | Zij gaf mij een lief meisje . | dative + adjective |
| I have never seen such a large dog. | Ik heb nog nooit zo'n grote hond gezien. | Ik heb nooit zulk een hond gezien . | negation + superlative |

## Summary

- sentences: 20
- empty outputs: 0
- outputs sharing >=50% of the reference words: 16 / 20
