# test_unseen (calibrated)

2997 sentences

| | intent | feature | quality | box | count | **all five** |
| --- | --- | --- | --- | --- | --- | --- |
| all (2997) | 94.3% | 99.9% | 95.4% | 97.9% | 98.7% | **87.9%** |
| en (999) | 96.9% | 100.0% | 99.1% | 99.9% | 98.5% | **94.6%** |
| es (999) | 92.5% | 100.0% | 99.6% | 97.2% | 99.9% | **89.8%** |
| eu (999) | 93.4% | 99.8% | 87.5% | 96.6% | 97.7% | **79.3%** |

## Is the percentage right?

| question | accuracy | mean confidence | calibration error (ECE) |
| --- | --- | --- | --- |
| intent | 94.3% | 95.3% | 0.020 |
| feature | 99.9% | 99.8% | 0.002 |
| quality | 95.4% | 97.3% | 0.024 |
| box | 97.9% | 97.3% | 0.008 |
| count | 98.7% | 98.6% | 0.007 |

Refusing the answers the model is least sure of (confidence = weakest question that matters):

| confidence at least | sentences kept | all five right among them |
| --- | --- | --- |
| 0% | 100.0% | 87.9% |
| 50% | 94.4% | 90.2% |
| 70% | 89.6% | 92.5% |
| 80% | 86.7% | 93.3% |
| 90% | 82.5% | 94.3% |
| 95% | 76.3% | 95.6% |
| 99% | 53.1% | 98.6% |

## The two boxes

1042 orders named the big or the white box. Sent to the wrong one of the two: **0** (0.00%). Read as another box or none: 31 (2.98%).

## Safety

- Sentences that must not move the robot read as an order: **5 of 912** (0.55%)
- stop recognised: 88 of 105

## Per intent (recognised)

| intent | recognised |
| --- | --- |
| ask | 96.4% (165) |
| both | 100.0% (225) |
| cancel | 90.9% (66) |
| chat | 89.1% (165) |
| hello | 82.7% (75) |
| move | 97.6% (780) |
| no | 97.3% (75) |
| pick | 90.3% (195) |
| refuse | 96.7% (120) |
| repeat | 100.0% (66) |
| reply | 97.8% (225) |
| sort | 92.7% (660) |
| stop | 83.8% (105) |
| yes | 80.0% (75) |

## Errors (363), first 40

- `eu` 'Gomendattu liburu bat.': intent: chat -> refuse (37%)
- `eu` 'Tira, gomendatu liburu bat.': intent: chat -> refuse (63%)
- `eu` 'tira, jaso ezazu txapa oso bat': intent: pick -> refuse (98%), quality: good -> any (100%), count: one -> none (99%)
- `es` 'a esa caja contenedora blanca, criba las piezas útiles que llevan perno ahora': intent: sort -> pick (89%)
- `eu` 'pernoa txertatuta duten txapa osoak: antolatu': quality: good -> bad (89%), box: none -> big (36%)
- `es` 'Necesito que ahora alces cualquier pieza que tiene perno ahora': intent: pick -> move (36%)
- `eu` 'Kaixo, kutxatxo zurian pernoa txertatuta duten hiru zati antolatu itzazu.': quality: any -> bad (98%)
- `en` 'reposition the satisfactory ones with a bolt in the whitish box': count: two -> all (41%)
- `eu` 'Kaixo, kutxatxo zuria polita iruditzen zait': intent: refuse -> reply (98%), box: none -> white (45%)
- `eu` 'kaixo, multzokatu pernorik jarrita ez duten pieza osoak kutxatxo zurira': quality: good -> any (99%)
- `eu` 'Ordenatu pernoa txertatuta duen eta matxurak dituen txapa bat': box: none -> big (49%)
- `es` 'todas las chapas en orden al contenedor blanquecino': box: white -> other (66%)
- `eu` 'eraman itzazu bi txapa kutxa zurira': intent: move -> refuse (53%), box: white -> none (48%), count: two -> none (86%)
- `eu` 'Pernoa txertatuta duten txapak kutxatxo tamaina handikoan multzokatu itzazu lehenbailehen': quality: any -> bad (97%)
- `es` 'Necesito que ahora cribes las dos chapas a ese receptáculo gigante': intent: sort -> pick (51%)
- `eu` 'txapak kutxa zurira sartu orain': intent: move -> refuse (85%)
- `en` 'I want you to ferry the ones with a defect in the white-colored receptacle': count: two -> all (44%)
- `eu` 'Pernorik gabeko zati oso guztiak': quality: good -> any (49%)
- `eu` 'eta orain pernoa txertatuta duen txapa kutxatxo kolore zurikora lekualdatu mesedez': quality: any -> bad (98%)
- `eu` 'ondo, eraman pernorik ez duen txapa oso bat ontzi kolore zurikora': quality: good -> any (95%)
- `es` 'criba las chapas buenas dentro de ese receptáculo blanco.': intent: sort -> pick (49%)
- `eu` 'beno, bi pieza baztertu kontenedore zurira kategorizatu': quality: bad -> any (94%)
- `eu` 'Bale, pernorik jarrita ez duten zati oso guztiak hartu itzazu eskerrik asko': quality: good -> any (53%)
- `es` 'Criba una chapa a la caja contenedora de color blanco, gracias': intent: sort -> refuse (81%), box: white -> none (47%), count: one -> none (49%)
- `eu` 'kaixo, pernoa txertatuta duen txapa oso bat multzokatu mesedez': quality: good -> bad (98%), box: none -> big (71%)
- `es` 'en ese contenedor voluminoso': box: big -> other (80%)
- `eu` 'txapak kutxatxo morera pasatu itzazu.': box: other -> white (98%)
- `eu` 'zenbaat denbora geratzen da?': intent: ask -> chat (94%)
- `en` 'you should snatch the ones featuring bolts': count: two -> all (41%)
- `eu` 'Hiru zati oso zurian multzokatu nahiko nituzke.': quality: good -> any (87%)
- `eu` 'Zati guztiak zurira aldatu ditzakezu?': intent: move -> sort (78%)
- `eu` 'Gomendatu liburu bat orain': intent: chat -> refuse (36%)
- `eu` 'ez egin kasurik eskerrik asko': intent: cancel -> no (38%)
- `eu` 'beno, gomendatu liburu bat': intent: chat -> refuse (56%)
- `eu` 'Perno duten txapa baztertuak sailkatu mesedez': box: none -> other (84%)
- `eu` 'pernorik jarrita ez duten txapa guztiak kaxa zurira aldatu ditzakezu?': intent: move -> sort (92%)
- `eu` 'e, pernoa txertatuta duten pieza horiek bereizi behar dituzu!': quality: any -> bad (96%)
- `en` 'well into the oragne one': box: other -> big (38%)
- `eu` 'Txapa baztertuak kutxatxo zurira taldekatu itzazu': quality: bad -> any (52%)
- `eu` 'Orain pernorik jarrita ez duen txapa oso bat tamaina handikora banatu mesedez ahal baduzu': quality: good -> any (96%)

## Without calibration (temperature 1)

| question | ECE before | ECE after |
| --- | --- | --- |
| intent | 0.049 | 0.020 |
| feature | 0.001 | 0.002 |
| quality | 0.043 | 0.024 |
| box | 0.015 | 0.008 |
| count | 0.011 | 0.007 |
