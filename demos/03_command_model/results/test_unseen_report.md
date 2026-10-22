# test_unseen.jsonl

1500 sentences, 232 ms per sentence

| | intent | items | place | sort_by | **command** |
| --- | --- | --- | --- | --- | --- |
| all | 92.1% | 49.3% | 83.5% | 100.0% | **40.7%** |
| en | 91.6% | 46.2% | 88.0% | 100.0% | **39.4%** |
| es | 91.2% | 46.2% | 83.4% | 100.0% | **37.0%** |
| eu | 93.4% | 55.6% | 79.2% | 100.0% | **45.8%** |

- stop caught: - (0/0)
- false action rate: 0.7% (2/270 sentences that must not move the robot)

| intent | recognised |
| --- | --- |
| answer_value | 98.3% (120) |
| give | 99.4% (180) |
| pick | 96.7% (240) |
| pick_and_place | 99.8% (540) |
| question | 98.0% (150) |
| sequence | 61.5% (270) |

## Errors (889)

- `es` 'coges la canica amarilla': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "canica", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "canica", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'Jaso azenario horiak eta sar ezazu otzaran': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "azenario", "color": "yellow", "quantity": "all", "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "azenario", "color": "yellow", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `es` 'al contenedor': wrong items, place  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "contenedor", "color": null, "quantity": 1, "position": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`
- `en` 'The purple bit': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "any", "name": "bit", "color": "purple", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "bit", "color": "purple", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'mesedez, hartu adreilu horia eta jar ezazu otzarara': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "adreilu", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "adreilu", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'Primero coge las azules y luego la tijera blanca y mételos a tu derecha': wrong intent, place  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": "all", "position": null}, {"kind": "other", "name": "tijera", "color": "white", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "tijera", "color": "blue", "quantity": "all", "position": null}, {"kind": "other", "name": "tijera", "color": "white", "quantity": 1, "position": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `es` 'guarda la canica roja sobre la nevera': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `eu` 'grapagailu horia otzaran sar ezazu': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "grapagailu", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "grapagailu", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'venga, coge la canica': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "canica", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "canica", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'fetch a carrot and move it onto the sofa, thank you': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "carrot", "color": null, "quantity": 1, "position": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "carrot", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'set the white marble on the left on the yellow table': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "marble", "color": "white", "quantity": 1, "position": "left"}], "place": {"kind": "table", "color": "yellow"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "marble", "color": "white", "quantity": 1, "position": "left"}], "place": {"kind": "table", "color": "yellow"}, "sort_by": "none"}`
- `en` 'to the left.': wrong items  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "left", "color": null}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": null, "quantity": 1, "position": "left"}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `en` 'would you drop two yellow marbles inside the fridge please.': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "marbles", "color": "yellow", "quantity": 2, "position": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "marbles", "color": "yellow", "quantity": 2, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `eu` 'Utzi esfera berde bat zure eskuinean': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`
- `en` 'okay, take the bricks': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "bricks", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "bricks", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `es` 'levanta el ladrillo verde y la grapadora y mételos en la canasta': wrong intent, items  
  expected `{"intent": "sequence", "items": [{"kind": "cube", "name": "ladrillo", "color": "green", "quantity": 1, "position": null}, {"kind": "other", "name": "grapadora", "color": null, "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "ladrillo", "color": "green", "quantity": 1, "position": null}, {"kind": "other", "name": null, "color": null, "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `es` 'pilla la canica verde': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "canica", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "canica", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'where is the black sphere?': wrong items  
  expected `{"intent": "question", "items": [{"kind": "ball", "name": "sphere", "color": "black", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [{"kind": "cube", "name": "sphere", "color": "black", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'kanika horia hartu eta sartu kaxan.': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "kanika", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "kanika", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `eu` 'Hartu esfera berdeak': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "esfera", "color": "green", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "cube", "name": "esfera", "color": "green", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `es` 'toma la roja y suéltalo en el contenedor rojo!': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `en` 'do you see the yellow bit?': wrong items  
  expected `{"intent": "question", "items": [{"kind": "any", "name": "bit", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [{"kind": "other", "name": "bit", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'Mesedez, hartu kanika hori bat': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "kanika", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "kanika", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'toma la canica morada': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "canica", "color": "purple", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "canica", "color": "purple", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'Okay, first fetch the green bit and then the blue sphere and put them to the right': wrong intent, items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": "bit", "color": "green", "quantity": 1, "position": null}, {"kind": "ball", "name": "sphere", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "bit", "color": "green", "quantity": 1, "position": null}, {"kind": "cube", "name": "sphere", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`
- `es` 'coge la canica rosa y guárdala en el contenedor amarillo': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": "pink", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "yellow"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": "pink", "quantity": 1, "position": null}], "place": {"kind": "other", "color": "yellow"}, "sort_by": "none"}`
- `en` 'Set on the right side two yellow bits.': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "bits", "color": "yellow", "quantity": 2, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bits", "color": "yellow", "quantity": 2, "position": "right"}, {"kind": "cube", "name": "bits", "color": "yellow", "quantity": 2, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`
- `eu` 'Esfera horia.': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "ball", "name": "esfera", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "cube", "name": "Esfera", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'pick up the red sphere and drop it in the crate please': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "sphere", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "sphere", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `en` 'right, pass me the scissors': wrong items  
  expected `{"intent": "give", "items": [{"kind": "other", "name": "scissors", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "other", "name": "scissors", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'berdea, gorria, berdea, zuria.': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "white", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": "berdea", "color": null, "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "white", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'bring me the brick, the green brick and the green bricks': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "cube", "name": "brick", "color": null, "quantity": 1, "position": null}, {"kind": "cube", "name": "brick", "color": "green", "quantity": 1, "position": null}, {"kind": "cube", "name": "bricks", "color": "green", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "other", "name": "brick", "color": "green", "quantity": 1, "position": null}, {"kind": "cube", "name": "brick", "color": "green", "quantity": 1, "position": null}, {"kind": "other", "name": "brick", "color": "green", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `es` 'coge el trozo y suéltalo dentro del contenedor': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "trozo", "color": null, "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "trozo", "color": null, "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `es` 'Pon verde, rojo, verde y amarillo.': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": "verde", "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'mete dentro de la canasta tres ladrillos rojos': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "ladrillos", "color": "red", "quantity": 3, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "ladrillos", "color": "red", "quantity": 3, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `eu` 'hartu kanika berdea eta sartu sofan.': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "kanika", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "kanika", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` '¿cuántas canicas hay?': wrong items  
  expected `{"intent": "question", "items": [{"kind": "ball", "name": "canicas", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [{"kind": "other", "name": "canicas", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `en` 'okay, stick into the black hamper two green ones': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "ones", "color": "green", "quantity": 2, "position": null}], "place": {"kind": "basket", "color": "black"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "ones", "color": "green", "quantity": 2, "position": null}], "place": {"kind": "other", "color": "black"}, "sort_by": "none"}`
- `en` 'Get me a purple marble.': wrong items  
  expected `{"intent": "give", "items": [{"kind": "ball", "name": "marble", "color": "purple", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "cube", "name": "marble", "color": "purple", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'Coge la canica naranja y guardala en el contenedor, por favor': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": "orange", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": "orange", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `es` 'Mete el trozo amarillo al contenedor azul!': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "trozo", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "trozo", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `eu` 'Ados, zuria otzaran utzi, mesedez': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": null, "color": "white", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": null, "color": "white", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'Mmm, ¿puedes meter la canica en el contenedor rojo?': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": null, "quantity": 1, "position": null}], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": null, "quantity": 1, "position": null}], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`
- `eu` 'bloke urdina hartu eta sartu mahai urdinaren gainera': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloke", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "table", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloke", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "table", "color": "red"}, "sort_by": "none"}`
- `eu` 'Heldu kanika urdina.': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "kanika", "color": "blue", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "kanika", "color": "blue", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` '¿cuántas esferas hay?': wrong items  
  expected `{"intent": "question", "items": [{"kind": "ball", "name": "esferas", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [{"kind": "other", "name": "esferas", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'Mesedez, hartu hiru esfera arrosa eta bloke beltza eta sartu otzarara': wrong intent, items, place  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "esfera", "color": "pink", "quantity": 3, "position": null}, {"kind": "cube", "name": "bloke", "color": "black", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "pink", "quantity": 3, "position": null}, {"kind": "cube", "name": "bloke", "color": "black", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'can you get a scissors?': wrong intent  
  expected `{"intent": "pick", "items": [{"kind": "other", "name": "scissors", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "other", "name": "scissors", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'pass me five black bits and the pink bit': wrong intent, items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": "bits", "color": "black", "quantity": 5, "position": null}, {"kind": "any", "name": "bit", "color": "pink", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "cube", "name": "bits", "color": "black", "quantity": 5, "position": null}, {"kind": "other", "name": null, "color": "pink", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'kanika urdina hartu eta jarri ezkerrera.': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "kanika", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "kanika", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `es` 'pon la canica amarilla en el contenedor amarillo por favor.': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "yellow"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "yellow"}, "sort_by": "none"}`
- `en` 'take all green bricks and leave them on your left.': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bricks", "color": "green", "quantity": "all", "position": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "bricks", "color": "green", "quantity": "all", "position": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `en` 'place into the crate the pink marbles': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "marbles", "color": "pink", "quantity": "all", "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "marbles", "color": "pink", "quantity": "all", "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `en` 'would you drop the purple bit inside the orange hamper?': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "bit", "color": "purple", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "orange"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "bit", "color": "purple", "quantity": 1, "position": null}], "place": {"kind": "other", "color": "orange"}, "sort_by": "none"}`
- `eu` 'jaso lau adreilu urdin': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "adreilu", "color": "blue", "quantity": 4, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "adreilu", "color": "blue", "quantity": 4, "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'Mesedez, hartu zati hori bat eta esfera arrosak': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": "zati", "color": "yellow", "quantity": 1, "position": null}, {"kind": "ball", "name": "esfera", "color": "pink", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null}, {"kind": "any", "name": "esfera", "color": "pink", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `es` 'me traes todos los naranjas por favor': wrong items  
  expected `{"intent": "give", "items": [{"kind": "any", "name": null, "color": "orange", "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "any", "name": "naranjas", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `eu` 'Urdina, laranja, urdina mesedez': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "orange", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": "laranja", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'Stick in the hamper two yellow bricks': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bricks", "color": "yellow", "quantity": 2, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "bricks", "color": "yellow", "quantity": 2, "position": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`
- `es` '¿dónde está la caja?': wrong items  
  expected `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [{"kind": "other", "name": "caja", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
