# test_v2_only.jsonl

38 sentences, 76 ms per sentence

| | intent | items | place | sort_by | **command** |
| --- | --- | --- | --- | --- | --- |
| all | 86.8% | 92.1% | 100.0% | 100.0% | **81.6%** |
| en | 88.9% | 100.0% | 100.0% | 100.0% | **88.9%** |
| es | 87.0% | 87.0% | 100.0% | 100.0% | **78.3%** |
| eu | 83.3% | 100.0% | 100.0% | 100.0% | **83.3%** |

- stop caught: 100.0% (2/2)
- false action rate: 0.0% (0/9 sentences that must not move the robot)

| intent | recognised |
| --- | --- |
| answer_no | 100.0% (1) |
| answer_value | 100.0% (1) |
| answer_yes | 100.0% (1) |
| cancel | 100.0% (1) |
| clear | 20.0% (5) |
| give | 100.0% (5) |
| greeting | 100.0% (1) |
| pick | 100.0% (3) |
| pick_and_place | 100.0% (1) |
| question | 100.0% (1) |
| repeat | 100.0% (1) |
| sequence | 83.3% (6) |
| sort | 100.0% (9) |
| stop | 100.0% (2) |

## Errors (7)

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
