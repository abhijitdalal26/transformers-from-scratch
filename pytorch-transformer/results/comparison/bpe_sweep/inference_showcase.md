# What the models actually say

Greedy decoding, same sentences for every checkpoint.

## Never-seen sentences (in neither train nor val)

| English | Reference | `bpe ep9` | `bpe ep10` | `bpe ep17` | `wordlevel ep17` |
|---|---|---|---|---|---|
| The cat sleeps on the table. | De kat slaapt op de tafel. | de kat was op de tafel. | de kat slaapt op de tafel. | de schrei van de tafel op de tafel. | De kat , die aan de tafel stond . |
| I drink coffee every morning. | Ik drink elke morgen koffie. | ik baadde den 11den dag. | ik heb den morgen al wat hij deed. | ik maakte een gereed om alles te schieten. | Ik drinken iedere . |
| She does not like fish. | Zij houdt niet van vis. | zij zijn niet zoo visschen. | zij zijn niet zoo visschen. | zij vischt het niet zooals wij weten. | Zij doet er niet toe . |
| We were very happy yesterday. | Wij waren gisteren zeer gelukkig. | het was gisteren gelukkig. | wij waren gelukkig. | het was gisteren al gelukkig, wij waren gisteren zeer gelukkig. | Wij waren gisteren al zeer gelukkig . |
| The book is on the shelf. | Het boek ligt op de plank. | het boek is op de borst gekomen. | het boek is op het boek gekomen. | het boek is op haar beurt. | Het boek ligt op den heuvel . |
| He is taller than his brother. | Hij is langer dan zijn broer. | hij is veel waard dan zijn broeder. | hij is veel waard dan zijn broer. | hij is veel grooter dan zijn broeder. | Hij is veel grooter dan zijn broer . |
| What are you doing? | Wat doe je? | wat doet gij toch? | wat doet gij? | wat heb je nou wel noodig? | Wat doet u dat ? |
| Where is the station? | Waar is het station? | waar is het station? | waar is het station? | waar is het station? | Waar is het station ? |
| I have two brothers and one sister. | Ik heb twee broers en een zuster. | ik heb twee broer en een zuster. | ik heb twee broer en een zuster ook. | ik heb twee broeders en een zuster logeeren. | Ik heb twee broeders en een zuster . |
| The weather is cold today. | Het weer is koud vandaag. | het weer is heden koud. | het weer is heden koud. | het weer is heden koud. | Het is vandaag koud . |
| They will come to the party. | Zij zullen naar het feest komen. | zij zullen den anderen kant wel eens komen. | zij zullen den anderen kant komen. | ze zullen wel hier te komen. | Zij zullen het aan iedereen komen . |
| This is a very interesting book. | Dit is een zeer interessant boek. | dat is een zeer belangwekkend boek. | dit is een zeer belangwekkend boek. | dit is een zeer belangwekkend werk. | Dit is een mooie boek . |
| The child is playing in the garden. | Het kind speelt in de tuin. | het kind is in den tuin. | het kind is in den tuin gekomen. | het kind zit in den tuin. | Het kind is in den tuin . |
| I do not understand this word. | Ik begrijp dit woord niet. | ik begrijp niet, dat dit woord er sprake was. | ik begrijp niet. | ik begrijp niet of ik dit woord nog wel begreep. | Ik begrijp het niet . |
| The man who came yesterday is my friend. | De man die gisteren kwam is mijn vriend. | de man die gisteren is aangekomen. | die man is gisteren aangekomen. | de man, die gisteren is geweest, is mijn vriend! | De man , die gisteren is , is reeds mijn vriend ! |
| Please open the window. | Open alsjeblieft het raam. | en wat het raam betreft. | als het raam openging. | betuigerlijk het raam open. | Het raam is goed . |
| The colour of the sky is blue. | De kleur van de hemel is blauw. | de kleur der hemel is een blauw fluit. | de kleur van het hemel is een blauw flink. | de ruhmkorff is een blauw flater. | De kleur van het uitspansel . |
| Water boils at one hundred degrees. | Water kookt bij honderd graden. | water uit honderd graden. | water uit honderd graad. | water der honderd graad! | Water uit honderd graad . |
| She gave me a beautiful gift. | Zij gaf mij een prachtig geschenk. | zij gaf mij een lief meisje aan te nemen. | zij gaf mij een lief, die een liefhad. | zij gaf mij een mooie aanstakigheid. | Zij gaf mij een lief meisje . |
| I have never seen such a large dog. | Ik heb nog nooit zo'n grote hond gezien. | ik heb nooit zoo'n hond nooit gezien. | ik heb nooit zoo'n hond gezien. | ik heb nog nooit zoo'n hond nooit gezien. | Ik heb nooit zulk een hond gezien . |

## Held-out sentences from the validation split

| English | Reference | `bpe ep9` | `bpe ep10` | `bpe ep17` | `wordlevel ep17` |
|---|---|---|---|---|---|
| But it was too late. | Doch het was te laat. | doch het was te laat. | maar het was te laat. | doch het was te laat. | Maar het was te laat . |
| Conseil had said the true word. | Koenraad had het juiste woord gekozen. | koenraad had het waar gezegd. | koenraad had het waar gezegd. | koenraad had het waar gezegd. | Koenraad had het juiste woord gezegd . |
| He would want to know all about it. | Niets zou hem kunnen tegenhouden! | hij zou er alles van weten. | hij zou er alles van weten komen. | hij zou alles hebben gewonnen. | Hij wilde alles vertellen , wat hij er van denk . |
| But she fled upstairs, nevertheless, with Sid and Mary at her heels. | Desniettemin vloog zij doodsbleek en met bevende lippen de trappen op en Sid en Marie achter haar aan. | doch zij verwijderde zich, terwijl marie en marie naast haar stond. | maar zij snelde naar de trap, doch bleef onbeweeglijk liggen, terwijl marie op den grond stond. | doch zij verwijderde zich haastig en marie en marie op den grond. | Maar zij sloot , met een enkelen en bevende , terwijl Marie haar zakdoek voor de deur wierp . |
| The entire day passed without my being honored by a visit from Captain Nemo. | De dag ging voorbij, zonder dat ik met een bezoek van den kapitein vereerd werd. | de dag ging voorbij zonder mijn bezoek te brengen, om van kapitein nemo te vernemen. | de dag ging voorbij zonder mijn bezoek van den kapitein van den kapitein af. | de dag ging voorbij, zonder dat ik mijn bezoek zou door den kapitein werd ontvolkt. | De dag brak den dag en ging van mijn bezoek af , door een bezoek van kapitein Nemo . |
| Joe took a pin out of his lapel and began to assist in exercising the prisoner. | Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje mores te leeren. | joe nam een spekpeltje en begon in de gevangenis te bevelen. | joe nam een spekje uit de kerk en begon in de gevangenis te gaan, om de gevangene in te redden. | joe haalde een spijker uit de gezondheid en begon in de gastvrijheid te wijken. | Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje te geven . |
