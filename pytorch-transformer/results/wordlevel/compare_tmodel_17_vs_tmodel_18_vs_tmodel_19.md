# Checkpoint comparison - tmodel_17 vs tmodel_18 vs tmodel_19

- **Scored on:** 60 identical held-out sentences (seed 42)
- **Decoding:** greedy | **device:** cpu

## Scores

| Metric | `tmodel_17` (ep 17) | `tmodel_18` (ep 18) | `tmodel_19` (ep 19) | Best |
|---|---|---|---|---|
| BLEU-4 | 26.3620 | 21.8280 | 21.1240 | `tmodel_17` |
| BLEU-1 prec | 0.5407 | 0.5236 | 0.4983 | `tmodel_17` |
| BLEU-2 prec | 0.2996 | 0.2732 | 0.2456 | `tmodel_17` |
| BLEU-3 prec | 0.2037 | 0.1685 | 0.1538 | `tmodel_17` |
| BLEU-4 prec | 0.1464 | 0.1118 | 0.1057 | `tmodel_17` |
| brevity pen | 1.0000 | 0.9579 | 1.0000 | `tmodel_17` |
| CER | 0.5560 | 0.5772 | 0.6054 | `tmodel_17` |
| WER | 0.9045 | 0.8997 | 0.9515 | `tmodel_18` |
| mean sent BLEU | 25.1980 | 22.7090 | 20.5710 | `tmodel_17` |
| median sent BLEU | 19.7900 | 14.5700 | 11.6110 | `tmodel_17` |
| sents with BLEU>0 | 39 | 38 | 34 | `tmodel_17` |
| exact match | 0 | 0 | 0 | `tmodel_17` |
| mean gen len | 24.2500 | 22.7700 | 24.2500 | - |

## Overall

Wins per directed metric: `tmodel_17` 11, `tmodel_18` 1, `tmodel_19` 0

**Winner: `tmodel_17`**

## Head-to-head per sentence

Sentences where each checkpoint had the best per-sentence BLEU:

| | count |
|---|---|
| `tmodel_17` best | 38 |
| `tmodel_18` best | 11 |
| `tmodel_19` best | 11 |
| tied across checkpoints | 21 |

## Side-by-side translations

### 1

- **SOURCE:** The China, for Liverpool, had started three-quarters of an hour before!
- **REFERENCE:** De China was vijf en veertig minuten te voren naar Liverpool vertrokken.
- **`tmodel_17`:** De China was vijf en veertig minuten vroeger voorbij .
- **`tmodel_18`:** De China was vijf en veertig minuten te voren naar Liverpool vertrokken .
- **`tmodel_19`:** De China was te Liverpool , de China , twee uur geleden , de van de wereld !

### 2

- **SOURCE:** He will give you a horse all saddled, and will point out to you the road you ought to take. You will find, in the same way, four relays on your route.
- **REFERENCE:** Hij zal u dan een gezadeld paard geven en u den weg wijzen, dien gij volgen moet; gij zult vier wisselplaatsen vinden.
- **`tmodel_17`:** Hij zal u een paard geven en den weg naar u geleiden ; gij zult er vier op vinden , zoodat wij vier op uw hoede zijn .
- **`tmodel_18`:** Hij zal u een paard geven en u den weg naar u zenden ; gij zult u d��r doen vinden , gij zult vier vinden .
- **`tmodel_19`:** Hij zal u een gezadeld paard geven en u den weg wijzen , dien gij volgen moet ; gij zult vier vinden .

### 3

- **SOURCE:** Joe took a pin out of his lapel and began to assist in exercising the prisoner.
- **REFERENCE:** Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje mores te leeren.
- **`tmodel_17`:** Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje te geven .
- **`tmodel_18`:** Joe nam een speld uit den lompen en begon zich te willen onderwerpen , om aan de te straffen .
- **`tmodel_19`:** Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden , om het gezag te geven , om den gevangene te doen ontvangen .

### 4

- **SOURCE:** Conseil had said the true word.
- **REFERENCE:** Koenraad had het juiste woord gekozen.
- **`tmodel_17`:** Koenraad had het juiste woord gezegd .
- **`tmodel_18`:** Koenraad had het juiste woord gezegd .
- **`tmodel_19`:** Koenraad had het woord gezegd .

### 5

- **SOURCE:** About this time the two Miss Steeles, lately arrived at their cousin's house in Bartlett's Buildings, Holburn, presented themselves again before their more grand relations in Conduit and Berkeley Streets; and were welcomed by them all with great cordiality.
- **REFERENCE:** Omstreeks dezen tijd kwamen de dames Steele, die reeds een poosje waren gelogeerd bij hun neef in Bartlett's Buildings, Holborn, zich weer vertoonen bij hun deftiger verwanten in Conduit en Berkeley Street; en zij werden door allen bijzonder hartelijk ontvangen.
- **`tmodel_17`:** Omstreeks dezen tijd kwamen de dames Steele , die reeds een poosje waren gelogeerd bij hun neef in Bartlett ' s Buildings , Holborn , zich weer vertoonen bij hun verwanten in Conduit en Berkeley Street ; en zij werden door alle mogelijke vriendelijkheid aan hen ontvangen .
- **`tmodel_18`:** Omstreeks dezen tijd kwamen de dames Steele , die onlangs bij hun neef in Bartlett ' s Buildings , , reeds haastig , hun en Berkeley Street hadden hervat ; zij waren door hen en elkaar met veel vriendelijkheid te toonen .
- **`tmodel_19`:** Omstreeks dezen tijd kwamen de beide dames Steele uit hun neef , die te _Ami�ns_ hadden ontvangen , zich van de , reeds bezig hielden in hun en in Berkeley Street ; en werden reeds de kennis gegeven door al de van hen .

### 6

- **SOURCE:** The entire day passed without my being honored by a visit from Captain Nemo.
- **REFERENCE:** De dag ging voorbij, zonder dat ik met een bezoek van den kapitein vereerd werd.
- **`tmodel_17`:** De dag brak den dag en ging van mijn bezoek af , door een bezoek van kapitein Nemo .
- **`tmodel_18`:** De dag ging voorbij , zonder dat ik een bezoek van kapitein Nemo als .
- **`tmodel_19`:** De dag brak aan zonder dat ik een bezoek van den kapitein op de hoogte van kapitein was .

### 7

- **SOURCE:** "What do you say?"
- **REFERENCE:** --�Wat zegt gij?�
- **`tmodel_17`:** " Wat zegt gij ?"
- **`tmodel_18`:** --� Wat zegt gij ?�
- **`tmodel_19`:** " Wat zegt gij daar ?"

### 8

- **SOURCE:** I do not like crooked, twisted, blasted trees. I admire them much more if they are tall, straight, and flourishing.
- **REFERENCE:** Ik houd _niet_ van kromme, verdraaide, half vergane boomen; ik vind ze veel mooier als ze recht en hoog zijn en door en door gezond.
- **`tmodel_17`:** Ik houd van _niet_ van mijn kant , van half vergane boomen ; ik vind ze veel mooier als ze en door het zeewater zijn .
- **`tmodel_18`:** Ik houd van _niet_ van mijn , half boomen , als ze veel mooier zijn .
- **`tmodel_19`:** Ik houd niet van mijn , , , , dat zij zwaar zijn , en recht , zooals ze recht hebben .

### 9

- **SOURCE:** Let us look about: look about!"
- **REFERENCE:** Laten wij zoeken! laten wij zoeken!"
- **`tmodel_17`:** Laat ons zien : zie eens , zie eens hier !"
- **`tmodel_18`:** Laat ons zien : zie eens , zie eens !"
- **`tmodel_19`:** Laten wij zoeken !"

### 10

- **SOURCE:** His facial features, usually so emotionless, revealed a certain uneasiness.
- **REFERENCE:** Zijn gelaat, dat gewoonlijk zoo kalm was, drukte nu eenige ongerustheid uit.
- **`tmodel_17`:** Zijn gelaat , dat gewoonlijk zoo kalm was , drukte nu eenige ongerustheid uit .
- **`tmodel_18`:** Zijn gelaat , dat gewoonlijk zoo kalm was , drukte hem eenige ongerustheid omtrent de tafel .
- **`tmodel_19`:** Zijn gelaat , dat gewoonlijk zoo kalm was , was een buitengewone ongerustheid .

### 11

- **SOURCE:** "How, in the wrong?" said Athos.
- **REFERENCE:** --�Hoe, ongelijk!� riep Athos.
- **`tmodel_17`:** --� Hoe , ongelijk !� zeide Athos .
- **`tmodel_18`:** --� Hoe , ongelijk ?� vroeg Athos .
- **`tmodel_19`:** --� Hoe , ongelijk ?� vroeg Athos .

### 12

- **SOURCE:** Come, my friend, come, let me embrace you.
- **REFERENCE:** Kom, mijn vriend, kom, laat ik u omhelzen.
- **`tmodel_17`:** Kom , vriend , kom , laat mij omhelzen .
- **`tmodel_18`:** Kom , mijn vriend , kom , laat mij u omhelzen .
- **`tmodel_19`:** Kom , mijn vriend , kom , laat mij u omhelzen .

### 13

- **SOURCE:** Just then Captain Nemo and his chief officer appeared.
- **REFERENCE:** Op dat oogenblik verscheen de kapitein met zijn eersten stuurman.
- **`tmodel_17`:** Op dat oogenblik verscheen kapitein Nemo en zijn stuurman .
- **`tmodel_18`:** Op dat oogenblik kwam kapitein Nemo en zijn stuurman binnen .
- **`tmodel_19`:** Op dat oogenblik verschenen kapitein Nemo en zijn stuurman .

### 14

- **SOURCE:** The stranger cried that he acknowledged no other king but his Eminence.
- **REFERENCE:** De vreemdeling riep uit, dat hij geen anderen koning kende dan Zijne Eminentie.
- **`tmodel_17`:** De onbekende riep hem toe , dat hij geen anderen koning kende , maar Zijne Eminentie .
- **`tmodel_18`:** De onbekende riep , dat hij niemand hem kende dan Zijne Eminentie .
- **`tmodel_19`:** De onbekende riep uit , dat hij geen anderen koning kende , maar Zijne Eminentie .

### 15

- **SOURCE:** "Because, as Porthos was talking louder than the rest of us, he took him for the chief," said d�Artagnan.
- **REFERENCE:** --�Omdat hij Porthos, die meer sprak dan wij allen, als het hoofd beschouwde,� zeide d�Artagnan.
- **`tmodel_17`:** --� Omdat Porthos , die meer sprak dan de overige van ons beiden , liet hij den mijnen zeggen ,� zeide d � Artagnan .
- **`tmodel_18`:** --� Omdat Porthos meer sprak dan over ons beiden , veel meer van ons beiden , heeft hem voor den eersten dag ,� zeide d � Artagnan .
- **`tmodel_19`:** --� Omdat , zooals Porthos gepraat had , meer over het overige van ons gesprek , heeft hij hem gezegd , dat hij den eersten van zijn kant was .

### 16

- **SOURCE:** Louis XIII, like every weak mind, was wanting in generosity.
- **REFERENCE:** Lodewijk XIII, zooals alle zwakke geesten, ontbrak het aan edelmoedigheid.
- **`tmodel_17`:** Lodewijk XIII , zooals alle tengevolge van geest , ontbrak het aan de edelmoedigheid dezer vrouw .
- **`tmodel_18`:** Lodewijk XIII , zooals gewoonlijk , deed al de van zijn denkwijze schijnen .
- **`tmodel_19`:** Lodewijk XIII , zooals men , deed zich in de hoogste mate , wat men zegt .

### 17

- **SOURCE:** But she fled upstairs, nevertheless, with Sid and Mary at her heels.
- **REFERENCE:** Desniettemin vloog zij doodsbleek en met bevende lippen de trappen op en Sid en Marie achter haar aan.
- **`tmodel_17`:** Maar zij sloot , met een enkelen en bevende , terwijl Marie haar zakdoek voor de deur wierp .
- **`tmodel_18`:** Maar zij klom de trap op , Sid en Marie achter haar .
- **`tmodel_19`:** Maar zij klom de trap in , echter met Sid en Marie achter haar .

### 18

- **SOURCE:** The captain stared at me, crossing his arms.
- **REFERENCE:** De kapitein keek mij aan, terwijl hij de armen over elkander sloeg.
- **`tmodel_17`:** De kapitein keek mij aan , terwijl hij mij de armen over elkander vestigde .
- **`tmodel_18`:** De kapitein keek mij aan , terwijl hij mij de armen over elkander vestigde .
- **`tmodel_19`:** De kapitein keek mij aan , terwijl hij mij omkeerde , om hem te zien .

### 19

- **SOURCE:** Henry IV, when besieging Paris, had loaves and provisions thrown over the walls. The cardinal had little notes thrown over in which he represented to the Rochellais how unjust, selfish, and barbarous was the conduct of their leaders.
- **REFERENCE:** Hendrik IV, _Parijs_ belegerende, liet brood en mondbehoeften over de muren werpen.
- **`tmodel_17`:** Hendrik IV , _Parijs_ , en mondbehoeften over de muren ; de kardinaal had weinig geestdrift gezien , hetgeen hij met de lieden deed , waarmede hij behoorde , en van welke hij de deuren der belegerden genoot .
- **`tmodel_18`:** Hendrik IV , _Parijs_ , en voorraad bestaande uit de muren ; de muren hadden weinig aanteekeningen gemaakt ; hij had eenige aanteekeningen gemaakt in de lieden , waarin hij de belegerden eens en was .
- **`tmodel_19`:** Hendrik IV , - en la Chesnaye , en er waren grond door de muren gehad ; de kardinaal had eenige aanteekeningen gemaakt , om de belegerden van _la Rochelle_ te bederven , en hij was bezig de van hun standpunt en met hun standpunt , met een aantal vrouwen van de deuren te onderscheiden .

### 20

- **SOURCE:** Besides the dog, there was nothing left in the ship that had life; nor any goods, that I could see, but what were spoiled by the water.
- **REFERENCE:** Behalve den hond was er geen levend schepsel aan boord, en voor zoo verre ik zien kon, geene goederen, die niet door het water bedorven waren.
- **`tmodel_17`:** Behalve den hond was er niets in het vaartuig , dat zoo goed ik kon , en alle goederen , die ik niet zien kon , maar welke het water door het water bedorven waren .
- **`tmodel_18`:** Bovendien wist ik niet hoe het met den hond uit het schip was gegaan , en er eenige goederen in zou zien , wat ik door het water bedorven was .
- **`tmodel_19`:** Bovendien kon ik niet meer gelooven , dat het schip noch goederen , noch goederen , die ik zien kon , maar welke ik door het water bedorven was .
