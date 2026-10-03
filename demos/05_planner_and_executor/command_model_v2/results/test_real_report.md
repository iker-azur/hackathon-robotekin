# test_real.jsonl

136 sentences, 55 ms per sentence

| | intent | items | place | sort_by | **command** |
| --- | --- | --- | --- | --- | --- |
| all | 94.9% | 90.4% | 95.6% | 100.0% | **84.6%** |
| en | 95.0% | 95.0% | 95.0% | 100.0% | **85.0%** |
| es | 93.3% | 85.0% | 95.0% | 100.0% | **80.0%** |
| eu | 97.2% | 94.4% | 97.2% | 100.0% | **91.7%** |

- stop caught: 100.0% (7/7)
- false action rate: 0.0% (0/54 sentences that must not move the robot)

| intent | recognised |
| --- | --- |
| answer_no | 100.0% (2) |
| answer_value | 100.0% (7) |
| answer_yes | 100.0% (4) |
| cancel | 100.0% (4) |
| clear | 63.6% (11) |
| give | 100.0% (11) |
| greeting | 100.0% (4) |
| not_doable | 100.0% (9) |
| off_topic | 100.0% (6) |
| pick | 100.0% (9) |
| pick_and_place | 90.5% (21) |
| question | 100.0% (7) |
| repeat | 100.0% (4) |
| sequence | 93.3% (15) |
| sort | 100.0% (15) |
| stop | 100.0% (7) |

## Errors (21)

- `es` 'Coge la esfera azul y déjala en la canasta verde': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none"}`
- `es` 'No cojas la bola roja, coge la verde': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "bola", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` 'Mueve el bloque rojo a la derecha': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloque", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloque", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `es` 'Coge una bola roja y dos azules': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 2, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "bola", "color": "blue", "quantity": 2, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` 'Recoge la mesa, por favor': wrong items  
  expected `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`  
  got `{"intent": "clear", "items": [{"kind": "other", "name": "mesa", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` 'Pon rojo, azul y rojo en la caja': wrong items, place  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": {"kind": "box", "color": "red"}}], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`
- `es` 'Cogen el objeto verde y dejaren el incesto rojo': wrong intent, items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "objeto", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "other", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` 'Grab the green ball and toss it in the crate': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "ball", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "ball", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`
- `en` 'Take the red sphere and place it in the basket': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "sphere", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "sphere", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `en` 'I need the yellow cube in the green box': wrong intent  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "cube", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "cube", "name": "cube", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`
- `en` 'On the red tray': wrong place  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "tray", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `en` 'First the yellow ball, then the red one': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "yellow", "quantity": 1, "position": null}, {"kind": "any", "name": "one", "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "ball", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` 'Hartu esfera urdina eta jarri otzara gorrian': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "other", "color": "green"}, "sort_by": "none"}`
- `eu` 'gorria, urdina, gorria': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` 'quita las bolas rojas de la mesa y ponlas en la caja azul': wrong intent  
  expected `{"intent": "clear", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `en` 'take the red balls off the table and put them in the blue box': wrong intent  
  expected `{"intent": "clear", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "balls", "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `eu` 'kendu pilota gorriak mahaitik eta jarri kutxa urdinean': wrong intent  
  expected `{"intent": "clear", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "pilota", "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `es` 'coge las dos bolas rojas y ponlas en la caja azul': wrong intent  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": 2, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": 2, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `es` 'recoge la mesa': wrong items  
  expected `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`  
  got `{"intent": "clear", "items": [{"kind": "other", "name": "mesa", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` 'mete todo en la caja azul': wrong intent, items  
  expected `{"intent": "clear", "items": [], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "any", "name": null, "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `es` 'la de la izquierda': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": null, "quantity": 1, "position": "left", "place": null}], "place": null, "sort_by": "none"}`
