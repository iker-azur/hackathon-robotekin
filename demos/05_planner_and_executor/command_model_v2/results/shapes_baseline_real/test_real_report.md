# test_real.jsonl

136 sentences, 119 ms per sentence

| | intent | items | place | sort_by | **command** |
| --- | --- | --- | --- | --- | --- |
| all | 94.9% | 90.4% | 87.5% | 100.0% | **76.5%** |
| en | 95.0% | 95.0% | 85.0% | 100.0% | **77.5%** |
| es | 93.3% | 85.0% | 86.7% | 100.0% | **71.7%** |
| eu | 97.2% | 94.4% | 91.7% | 100.0% | **83.3%** |

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

## Errors (32)

- `es` 'Coge la esfera azul y déjala en la canasta verde': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.984, "place_kind": 0.413, "place_color": 0.945, "sort_by": 1.0}}`
- `es` 'No cojas la bola roja, coge la verde': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "bola", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.781, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Mueve el bloque rojo a la derecha': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloque", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bloque", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "left", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.965, "place_kind": 0.874, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Coge una bola roja y dos azules': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 2, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "bola", "color": "blue", "quantity": 2, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Recoge la mesa, por favor': wrong items  
  expected `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`  
  got `{"intent": "clear", "items": [{"kind": "other", "name": "mesa", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Pon rojo, azul y rojo en la caja': wrong items, place  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": {"kind": "box", "color": "red"}}], "place": {"kind": "box", "color": "red"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Cogen el objeto verde y dejaren el incesto rojo': wrong intent, items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "objeto", "color": "green", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "other", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.692, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'Take the red sphere and place it in the basket': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "sphere", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "sphere", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'I need the yellow cube in the green box': wrong intent  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "cube", "color": "yellow", "quantity": 1, "position": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "cube", "name": "cube", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.999, "place_color": 0.929, "sort_by": 1.0}}`
- `en` 'On the red tray': wrong place  
  expected `{"intent": "answer_value", "items": [], "place": {"kind": "tray", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 0.664, "sort_by": 1.0}}`
- `en` 'First the yellow ball, then the red one': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "yellow", "quantity": 1, "position": null}, {"kind": "any", "name": "one", "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "ball", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'Hartu esfera urdina eta jarri otzara gorrian': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": "red"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.998, "place_kind": 0.69, "place_color": 0.474, "sort_by": 1.0}}`
- `eu` 'gorria, urdina, gorria': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'mete las bolas rojas en la caja roja y las azules en la bandeja azul': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "ball", "name": null, "color": "blue", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "blue"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "ball", "name": "bolas", "color": "blue", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "blue"}}], "place": {"kind": "tray", "color": "blue"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.946, "place_kind": 0.748, "place_color": 0.526, "sort_by": 1.0}}`
- `es` 'pon los cubos verdes en la bandeja verde y los rojos en el cesto rojo': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "cube", "name": null, "color": "green", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "green"}}, {"kind": "cube", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "red"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "cube", "name": "cubos", "color": "green", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "green"}}, {"kind": "cube", "name": "cubos", "color": "red", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "red"}}], "place": {"kind": "basket", "color": "red"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.987, "place_kind": 0.999, "place_color": 0.779, "sort_by": 1.0}}`
- `es` 'separa por colores: las piezas azules dentro de la caja azul y las rojas en la bandeja roja': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": "all", "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "any", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "red"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "any", "name": "piezas", "color": "blue", "quantity": "all", "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "any", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "red"}}], "place": {"kind": "tray", "color": "red"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.728, "place_color": 0.867, "sort_by": 1.0}}`
- `en` 'put the red balls in the red box and the blue ones in the blue tray': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "ball", "name": null, "color": "blue", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "blue"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "ball", "name": "balls", "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "ball", "name": "balls", "color": "blue", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "blue"}}], "place": {"kind": "tray", "color": "blue"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.998, "place_kind": 0.559, "place_color": 0.978, "sort_by": 1.0}}`
- `en` 'sort the green cubes in the green basket and the red cubes into the red tray': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "cube", "name": null, "color": "green", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "green"}}, {"kind": "cube", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "red"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "cube", "name": "cubes", "color": "green", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "green"}}, {"kind": "cube", "name": "cubes", "color": "red", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "red"}}], "place": {"kind": "tray", "color": "red"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.926, "place_kind": 0.642, "place_color": 0.952, "sort_by": 0.963}}`
- `eu` 'sartu pilota gorriak kutxa gorrian eta pilota urdinak erretilu urdinean': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "ball", "name": null, "color": "blue", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "blue"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "ball", "name": "pilota", "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "ball", "name": null, "color": "blue", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "blue"}}], "place": {"kind": "tray", "color": "blue"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.94, "place_kind": 0.861, "place_color": 0.501, "sort_by": 1.0}}`
- `eu` 'jarri pilota berdeak saski berdean eta pilota gorriak kutxa gorrian': wrong place  
  expected `{"intent": "sort", "items": [{"kind": "ball", "name": null, "color": "green", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "green"}}, {"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}], "place": null, "sort_by": "color"}`  
  got `{"intent": "sort", "items": [{"kind": "ball", "name": "pilota", "color": "green", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "green"}}, {"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": {"kind": "box", "color": "red"}}], "place": {"kind": "box", "color": "red"}, "sort_by": "color", "sort_filter": [], "confidences": {"intent": 0.997, "place_kind": 0.7, "place_color": 0.45, "sort_by": 0.997}}`
- `es` 'pon el cubo rojo en la caja azul y la bola verde en la bandeja roja': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "ball", "name": null, "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "red"}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "ball", "name": "bola", "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "red"}}], "place": {"kind": "tray", "color": "red"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.993, "place_kind": 0.747, "place_color": 0.895, "sort_by": 1.0}}`
- `es` 'mete la bola roja en la caja azul y el cubo verde en el cesto verde': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "cube", "name": null, "color": "green", "quantity": 1, "position": null, "place": {"kind": "basket", "color": "green"}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "cube", "name": "cubo", "color": "green", "quantity": 1, "position": null, "place": {"kind": "basket", "color": "green"}}], "place": {"kind": "basket", "color": "green"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.959, "place_kind": 1.0, "place_color": 0.882, "sort_by": 1.0}}`
- `en` 'put the red cube in the blue box and the green ball in the red tray': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "ball", "name": null, "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "red"}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "cube", "name": "cube", "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "ball", "name": null, "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "red"}}], "place": {"kind": "tray", "color": "red"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.587, "place_color": 0.854, "sort_by": 1.0}}`
- `en` 'place the blue ball in the red box and the red cube in the green tray': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": null, "color": "blue", "quantity": 1, "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "green"}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "blue", "quantity": 1, "position": null, "place": {"kind": "box", "color": "red"}}, {"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "green"}}], "place": {"kind": "tray", "color": "green"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.998, "place_kind": 0.637, "place_color": 0.683, "sort_by": 1.0}}`
- `eu` 'jarri kubo gorria kutxa urdinean eta pilota berdea erretilu gorrian': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "ball", "name": null, "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "red"}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "cube", "name": null, "color": "red", "quantity": 1, "position": null, "place": {"kind": "box", "color": "blue"}}, {"kind": "ball", "name": null, "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": "red"}}], "place": {"kind": "tray", "color": "red"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.997, "place_kind": 0.767, "place_color": 0.356, "sort_by": 1.0}}`
- `es` 'quita las bolas rojas de la mesa y ponlas en la caja azul': wrong intent  
  expected `{"intent": "clear", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.997, "place_kind": 0.999, "place_color": 0.89, "sort_by": 1.0}}`
- `en` 'take the red balls off the table and put them in the blue box': wrong intent, place  
  expected `{"intent": "clear", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "balls", "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "table", "color": "blue"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.996, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'kendu pilota gorriak mahaitik eta jarri kutxa urdinean': wrong intent  
  expected `{"intent": "clear", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "pilota", "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.965, "place_kind": 1.0, "place_color": 0.79, "sort_by": 1.0}}`
- `es` 'coge las dos bolas rojas y ponlas en la caja azul': wrong intent  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": null, "color": "red", "quantity": 2, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": 2, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.997, "place_kind": 1.0, "place_color": 0.943, "sort_by": 1.0}}`
- `es` 'recoge la mesa': wrong items  
  expected `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`  
  got `{"intent": "clear", "items": [{"kind": "other", "name": "mesa", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'mete todo en la caja azul': wrong intent, items  
  expected `{"intent": "clear", "items": [], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "any", "name": null, "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.96, "place_kind": 1.0, "place_color": 0.937, "sort_by": 1.0}}`
- `es` 'la de la izquierda': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": null, "quantity": 1, "position": "left", "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
