# Laya `demos/03_command_model/models/laya-commands` on test_real.jsonl

98 sentences, 40 ms per sentence

| intent | place_kind | place_color | sort_by | **all four** |
| --- | --- | --- | --- | --- |
| 99.0% | 95.9% | 99.0% | 100.0% | **93.9%** |

all four right per language: en 96.8%, es 89.2%, eu 96.7%
- stop caught: 100.0% (5/5)
- false action rate: 0.0% (0/45)

## Errors (6)

- 'Coge la esfera azul y déjala en la canasta verde': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'none'}` got `{'intent': 'pick_and_place', 'place_kind': 'other', 'place_color': 'green', 'sort_by': 'none'}`
- 'No cojas la bola roja, coge la verde': want `{'intent': 'pick', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'La de la izquierda': want `{'intent': 'answer_value', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'answer_value', 'place_kind': 'left', 'place_color': 'none', 'sort_by': 'none'}`
- 'Cogen el objeto verde y dejaren el incesto rojo': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}` got `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'none', 'sort_by': 'none'}`
- 'Grab the green ball and toss it in the crate': want `{'intent': 'pick_and_place', 'place_kind': 'box', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'pick_and_place', 'place_kind': 'other', 'place_color': 'none', 'sort_by': 'none'}`
- 'Hartu esfera urdina eta jarri otzara gorrian': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}` got `{'intent': 'pick_and_place', 'place_kind': 'other', 'place_color': 'red', 'sort_by': 'none'}`
