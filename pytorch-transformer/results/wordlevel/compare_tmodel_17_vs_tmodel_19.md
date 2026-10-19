# Checkpoint comparison - `tmodel_17` vs `tmodel_19`

- **Scored on:** 60 identical held-out sentences (seed 42)
- **Decoding:** greedy | **device:** cpu

## Scores

| Metric | `tmodel_17` (ep 17) | `tmodel_19` (ep 19) | Winner |
|---|---|---|---|
| **BLEU-4** | 26.36 | 21.12 | `tmodel_17` |
| BLEU-1 precision | 0.5407 | 0.4983 | `tmodel_17` |
| BLEU-2 precision | 0.2996 | 0.2456 | `tmodel_17` |
| BLEU-3 precision | 0.2037 | 0.1538 | `tmodel_17` |
| BLEU-4 precision | 0.1464 | 0.1057 | `tmodel_17` |
| CER (lower better) | 0.5560 | 0.6054 | `tmodel_17` |
| WER (lower better) | 0.9045 | 0.9515 | `tmodel_17` |
| exact match | 0 | 0 | tie |
| mean sentence BLEU | 25.20 | 20.57 | `tmodel_17` |
| sentences with BLEU>0 | 39 | 34 | `tmodel_17` |

## Head-to-head per sentence

| | count |
|---|---|
| `tmodel_17` better | 26 |
| `tmodel_19` better | 16 |
| identical | 18 |

## Side-by-side translations

### 1

- **SOURCE:** He will give you a horse all saddled, and will point out to you the road you ought to take. You will find, in the same way, four relays on your route.
- **REFERENCE:** Hij zal u dan een gezadeld paard geven en u den weg wijzen, dien gij volgen moet; gij zult vier wisselplaatsen vinden.
- **`tmodel_17`:** Hij zal u een paard geven en den weg naar u geleiden ; gij zult er vier op vinden , zoodat wij vier op uw hoede zijn .
- **`tmodel_19`:** Hij zal u een gezadeld paard geven en u den weg wijzen , dien gij volgen moet ; gij zult vier vinden .

### 2

- **SOURCE:** Conseil had said the true word.
- **REFERENCE:** Koenraad had het juiste woord gekozen.
- **`tmodel_17`:** Koenraad had het juiste woord gezegd .
- **`tmodel_19`:** Koenraad had het woord gezegd .

### 3

- **SOURCE:** About this time the two Miss Steeles, lately arrived at their cousin's house in Bartlett's Buildings, Holburn, presented themselves again before their more grand relations in Conduit and Berkeley Streets; and were welcomed by them all with great cordiality.
- **REFERENCE:** Omstreeks dezen tijd kwamen de dames Steele, die reeds een poosje waren gelogeerd bij hun neef in Bartlett's Buildings, Holborn, zich weer vertoonen bij hun deftiger verwanten in Conduit en Berkeley Street; en zij werden door allen bijzonder hartelijk ontvangen.
- **`tmodel_17`:** Omstreeks dezen tijd kwamen de dames Steele , die reeds een poosje waren gelogeerd bij hun neef in Bartlett ' s Buildings , Holborn , zich weer vertoonen bij hun verwanten in Conduit en Berkeley Street ; en zij werden door alle mogelijke vriendelijkheid aan hen ontvangen .
- **`tmodel_19`:** Omstreeks dezen tijd kwamen de beide dames Steele uit hun neef , die te _Ami�ns_ hadden ontvangen , zich van de , reeds bezig hielden in hun en in Berkeley Street ; en werden reeds de kennis gegeven door al de van hen .

### 4

- **SOURCE:** The China, for Liverpool, had started three-quarters of an hour before!
- **REFERENCE:** De China was vijf en veertig minuten te voren naar Liverpool vertrokken.
- **`tmodel_17`:** De China was vijf en veertig minuten vroeger voorbij .
- **`tmodel_19`:** De China was te Liverpool , de China , twee uur geleden , de van de wereld !

### 5

- **SOURCE:** I do not like crooked, twisted, blasted trees. I admire them much more if they are tall, straight, and flourishing.
- **REFERENCE:** Ik houd _niet_ van kromme, verdraaide, half vergane boomen; ik vind ze veel mooier als ze recht en hoog zijn en door en door gezond.
- **`tmodel_17`:** Ik houd van _niet_ van mijn kant , van half vergane boomen ; ik vind ze veel mooier als ze en door het zeewater zijn .
- **`tmodel_19`:** Ik houd niet van mijn , , , , dat zij zwaar zijn , en recht , zooals ze recht hebben .

### 6

- **SOURCE:** Let us look about: look about!"
- **REFERENCE:** Laten wij zoeken! laten wij zoeken!"
- **`tmodel_17`:** Laat ons zien : zie eens , zie eens hier !"
- **`tmodel_19`:** Laten wij zoeken !"

### 7

- **SOURCE:** His facial features, usually so emotionless, revealed a certain uneasiness.
- **REFERENCE:** Zijn gelaat, dat gewoonlijk zoo kalm was, drukte nu eenige ongerustheid uit.
- **`tmodel_17`:** Zijn gelaat , dat gewoonlijk zoo kalm was , drukte nu eenige ongerustheid uit .
- **`tmodel_19`:** Zijn gelaat , dat gewoonlijk zoo kalm was , was een buitengewone ongerustheid .

### 8

- **SOURCE:** "How, in the wrong?" said Athos.
- **REFERENCE:** --�Hoe, ongelijk!� riep Athos.
- **`tmodel_17`:** --� Hoe , ongelijk !� zeide Athos .
- **`tmodel_19`:** --� Hoe , ongelijk ?� vroeg Athos .

### 9

- **SOURCE:** Come, my friend, come, let me embrace you.
- **REFERENCE:** Kom, mijn vriend, kom, laat ik u omhelzen.
- **`tmodel_17`:** Kom , vriend , kom , laat mij omhelzen .
- **`tmodel_19`:** Kom , mijn vriend , kom , laat mij u omhelzen .

### 10

- **SOURCE:** Just then Captain Nemo and his chief officer appeared.
- **REFERENCE:** Op dat oogenblik verscheen de kapitein met zijn eersten stuurman.
- **`tmodel_17`:** Op dat oogenblik verscheen kapitein Nemo en zijn stuurman .
- **`tmodel_19`:** Op dat oogenblik verschenen kapitein Nemo en zijn stuurman .

### 11

- **SOURCE:** The entire day passed without my being honored by a visit from Captain Nemo.
- **REFERENCE:** De dag ging voorbij, zonder dat ik met een bezoek van den kapitein vereerd werd.
- **`tmodel_17`:** De dag brak den dag en ging van mijn bezoek af , door een bezoek van kapitein Nemo .
- **`tmodel_19`:** De dag brak aan zonder dat ik een bezoek van den kapitein op de hoogte van kapitein was .

### 12

- **SOURCE:** Joe took a pin out of his lapel and began to assist in exercising the prisoner.
- **REFERENCE:** Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje mores te leeren.
- **`tmodel_17`:** Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden om het diertje te geven .
- **`tmodel_19`:** Joe nam een speld uit de panden van zijn buisje en begon de behulpzame hand te bieden , om het gezag te geven , om den gevangene te doen ontvangen .

### 13

- **SOURCE:** "Because, as Porthos was talking louder than the rest of us, he took him for the chief," said d�Artagnan.
- **REFERENCE:** --�Omdat hij Porthos, die meer sprak dan wij allen, als het hoofd beschouwde,� zeide d�Artagnan.
- **`tmodel_17`:** --� Omdat Porthos , die meer sprak dan de overige van ons beiden , liet hij den mijnen zeggen ,� zeide d � Artagnan .
- **`tmodel_19`:** --� Omdat , zooals Porthos gepraat had , meer over het overige van ons gesprek , heeft hij hem gezegd , dat hij den eersten van zijn kant was .

### 14

- **SOURCE:** Louis XIII, like every weak mind, was wanting in generosity.
- **REFERENCE:** Lodewijk XIII, zooals alle zwakke geesten, ontbrak het aan edelmoedigheid.
- **`tmodel_17`:** Lodewijk XIII , zooals alle tengevolge van geest , ontbrak het aan de edelmoedigheid dezer vrouw .
- **`tmodel_19`:** Lodewijk XIII , zooals men , deed zich in de hoogste mate , wat men zegt .

### 15

- **SOURCE:** But she fled upstairs, nevertheless, with Sid and Mary at her heels.
- **REFERENCE:** Desniettemin vloog zij doodsbleek en met bevende lippen de trappen op en Sid en Marie achter haar aan.
- **`tmodel_17`:** Maar zij sloot , met een enkelen en bevende , terwijl Marie haar zakdoek voor de deur wierp .
- **`tmodel_19`:** Maar zij klom de trap in , echter met Sid en Marie achter haar .

### 16

- **SOURCE:** The captain stared at me, crossing his arms.
- **REFERENCE:** De kapitein keek mij aan, terwijl hij de armen over elkander sloeg.
- **`tmodel_17`:** De kapitein keek mij aan , terwijl hij mij de armen over elkander vestigde .
- **`tmodel_19`:** De kapitein keek mij aan , terwijl hij mij omkeerde , om hem te zien .

### 17

- **SOURCE:** Henry IV, when besieging Paris, had loaves and provisions thrown over the walls. The cardinal had little notes thrown over in which he represented to the Rochellais how unjust, selfish, and barbarous was the conduct of their leaders.
- **REFERENCE:** Hendrik IV, _Parijs_ belegerende, liet brood en mondbehoeften over de muren werpen.
- **`tmodel_17`:** Hendrik IV , _Parijs_ , en mondbehoeften over de muren ; de kardinaal had weinig geestdrift gezien , hetgeen hij met de lieden deed , waarmede hij behoorde , en van welke hij de deuren der belegerden genoot .
- **`tmodel_19`:** Hendrik IV , - en la Chesnaye , en er waren grond door de muren gehad ; de kardinaal had eenige aanteekeningen gemaakt , om de belegerden van _la Rochelle_ te bederven , en hij was bezig de van hun standpunt en met hun standpunt , met een aantal vrouwen van de deuren te onderscheiden .

### 18

- **SOURCE:** "The bodies were not plundered, were they? It appears the conquerors had something else to do."
- **REFERENCE:** --�Men heeft den tijd niet gehad hen uit te kleeden, niet waar? dewijl men voor het oogenblik iets meer gewichtigs te doen had.�
- **`tmodel_17`:** " Er waren geen tijd , die verloren waren , niet waar ? het schijnt , dat men iets anders te doen had ."
- **`tmodel_19`:** --� Men heeft hun niet gedaan , omdat men zich niet alleen tot den dienst bevond , en men schijnt nog iets anders te doen .�

### 19

- **SOURCE:** The stranger cried that he acknowledged no other king but his Eminence.
- **REFERENCE:** De vreemdeling riep uit, dat hij geen anderen koning kende dan Zijne Eminentie.
- **`tmodel_17`:** De onbekende riep hem toe , dat hij geen anderen koning kende , maar Zijne Eminentie .
- **`tmodel_19`:** De onbekende riep uit , dat hij geen anderen koning kende , maar Zijne Eminentie .

### 20

- **SOURCE:** I recall seeing the bay of El Tur completely red, like a lake of blood."
- **REFERENCE:** Ik herinner mij de golf van Suez bij Tor geheel roodgekleurd te hebben gezien, even alsof het een meer met bloed was."
- **`tmodel_17`:** Ik herinner mij de golf van Suez te hebben met een meer bloed , als het ware bloed ."
- **`tmodel_19`:** Ik heb de golf van Suez , zoo lief als een meer , dat zich met bloed naar het bloed uitstrekt ."
