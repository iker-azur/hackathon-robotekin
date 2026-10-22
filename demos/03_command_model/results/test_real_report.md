# test_real.jsonl

98 sentences, 176 ms per sentence

| | intent | items | place | sort_by | **command** |
| --- | --- | --- | --- | --- | --- |
| all | 95.9% | 87.8% | 94.9% | 100.0% | **85.7%** |
| en | 96.8% | 93.5% | 96.8% | 100.0% | **90.3%** |
| es | 91.9% | 78.4% | 94.6% | 100.0% | **78.4%** |
| eu | 100.0% | 93.3% | 93.3% | 100.0% | **90.0%** |

- stop caught: 100.0% (5/5)
- false action rate: 2.2% (1/45 sentences that must not move the robot)

| intent | recognised |
| --- | --- |
| answer_no | 100.0% (1) |
| answer_value | 100.0% (6) |
| answer_yes | 100.0% (3) |
| cancel | 100.0% (3) |
| clear | 100.0% (6) |
| give | 100.0% (6) |
| greeting | 100.0% (3) |
| not_doable | 88.9% (9) |
| off_topic | 100.0% (6) |
| pick | 100.0% (6) |
| pick_and_place | 85.0% (20) |
| question | 100.0% (6) |
| repeat | 100.0% (3) |
| sequence | 100.0% (9) |
| sort | 100.0% (6) |
| stop | 100.0% (5) |

## Errors (14)

- `es` 'Coge la esfera azul y déjala en la canasta verde': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none"}`
- `es` 'No cojas la bola roja, coge la verde': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": "bola", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'Mueve el bloque rojo a la derecha': wrong intent, items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloque", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "not_doable", "items": [], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `es` 'En la caja azul': wrong items  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": null, "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `es` '¿Qué ves en la mesa?': wrong items  
  expected `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [{"kind": "other", "name": null, "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'Tira la bola al suelo': wrong intent, items  
  expected `{"intent": "not_doable", "items": [{"kind": "ball", "name": "bola", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "bola", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` 'Pon rojo, azul y rojo en la caja': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `es` 'Cogen el objeto verde y dejaren el incesto rojo': wrong intent, items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "objeto", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": "incesto", "color": "green", "quantity": 1, "position": null}, {"kind": "other", "name": "incesto", "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` 'Take the red sphere and place it in the basket': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "sphere", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "sphere", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `en` 'I need the yellow cube in the green box': wrong intent  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "cube", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "cube", "name": "cube", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`
- `en` 'On the red tray': wrong items, place  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "tray", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "tray", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `eu` 'Hartu esfera urdina eta jarri otzara gorrian': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "other", "color": "green"}, "sort_by": "none"}`
- `eu` 'Kutxa gorrian': wrong place  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`
- `eu` 'gorria, urdina, gorria': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": "gorria", "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
