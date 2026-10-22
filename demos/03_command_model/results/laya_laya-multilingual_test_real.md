# Laya `convaiinnovations/laya-multilingual` on test_real.jsonl

98 sentences, 109 ms per sentence

| intent | place_kind | place_color | sort_by | **all four** |
| --- | --- | --- | --- | --- |
| 34.7% | 48.0% | 52.0% | 77.6% | **5.1%** |

all four right per language: en 9.7%, es 2.7%, eu 3.3%
- stop caught: 80.0% (4/5)
- false action rate: 6.7% (3/45)

## Errors (93)

- 'Coge la bola roja y métela en el cesto verde': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'none'}`
- 'Me puedes coger las peras y ponerlas en el cesto rojo': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}` got `{'intent': 'pick_and_place', 'place_kind': 'none', 'place_color': 'red', 'sort_by': 'none'}`
- 'coge el objeto verde y déjalo en la caja roja': want `{'intent': 'pick_and_place', 'place_kind': 'box', 'place_color': 'red', 'sort_by': 'none'}` got `{'intent': 'pick_and_place', 'place_kind': 'box', 'place_color': 'green', 'sort_by': 'color'}`
- 'Coge la esfera azul y déjala en la canasta verde': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'basket', 'place_color': 'blue', 'sort_by': 'color'}`
- 'No cojas la bola roja, coge la verde': want `{'intent': 'pick', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'stop', 'place_kind': 'right', 'place_color': 'green', 'sort_by': 'color'}`
- 'Coge el cubo azul, no, mejor el rojo': want `{'intent': 'pick', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'none', 'place_color': 'blue', 'sort_by': 'color'}`
- 'Mueve el bloque rojo a la derecha': want `{'intent': 'pick_and_place', 'place_kind': 'right', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'right', 'place_color': 'red', 'sort_by': 'color'}`
- '¿Qué tiempo hace hoy?': want `{'intent': 'off_topic', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'repeat', 'place_kind': 'none', 'place_color': 'blue', 'sort_by': 'none'}`
- 'Hola, buenas. ¿Qué tal estás?': want `{'intent': 'greeting', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'greeting', 'place_kind': 'table', 'place_color': 'none', 'sort_by': 'none'}`
- 'Pásame el destornillador': want `{'intent': 'give', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'clear', 'place_kind': 'other', 'place_color': 'none', 'sort_by': 'none'}`
- 'Coge una bola roja y dos azules': want `{'intent': 'sequence', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'other', 'place_color': 'blue', 'sort_by': 'color'}`
- 'Separa las piezas por colores': want `{'intent': 'sort', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'color'}` got `{'intent': 'sort', 'place_kind': 'other', 'place_color': 'none', 'sort_by': 'none'}`
- 'Recoge la mesa, por favor': want `{'intent': 'clear', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'repeat', 'place_kind': 'table', 'place_color': 'none', 'sort_by': 'none'}`
- '¡Para!': want `{'intent': 'stop', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'stop', 'place_kind': 'left', 'place_color': 'none', 'sort_by': 'none'}`
- 'Para, para, que se cae': want `{'intent': 'stop', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'stop', 'place_kind': 'left', 'place_color': 'black', 'sort_by': 'none'}`
- 'Otra vez': want `{'intent': 'repeat', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'repeat', 'place_kind': 'other', 'place_color': 'none', 'sort_by': 'none'}`
- 'Déjalo, no hace falta': want `{'intent': 'cancel', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'repeat', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'Sí, esa': want `{'intent': 'answer_yes', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'La de la izquierda': want `{'intent': 'answer_value', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'left', 'place_color': 'green', 'sort_by': 'none'}`
- 'En la caja azul': want `{'intent': 'answer_value', 'place_kind': 'box', 'place_color': 'blue', 'sort_by': 'none'}` got `{'intent': 'sequence', 'place_kind': 'box', 'place_color': 'blue', 'sort_by': 'none'}`
- '¿Qué ves en la mesa?': want `{'intent': 'question', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'question', 'place_kind': 'table', 'place_color': 'none', 'sort_by': 'none'}`
- '¿Dónde está el cubo amarillo?': want `{'intent': 'question', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'question', 'place_kind': 'none', 'place_color': 'yellow', 'sort_by': 'none'}`
- 'Tira la bola al suelo': want `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'repeat', 'place_kind': 'left', 'place_color': 'none', 'sort_by': 'none'}`
- 'Coge el coche': want `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'other', 'place_color': 'none', 'sort_by': 'none'}`
- 'Mete todos los cubos rojos en la bandeja': want `{'intent': 'pick_and_place', 'place_kind': 'tray', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'sequence', 'place_kind': 'tray', 'place_color': 'red', 'sort_by': 'none'}`
- 'Primero el cubo verde y luego la bola amarilla': want `{'intent': 'sequence', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'sequence', 'place_kind': 'none', 'place_color': 'yellow', 'sort_by': 'color'}`
- 'Pon rojo, azul y rojo en la caja': want `{'intent': 'sequence', 'place_kind': 'box', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'sequence', 'place_kind': 'box', 'place_color': 'red', 'sort_by': 'color'}`
- 'Dame la pelota más cercana': want `{'intent': 'give', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'give', 'place_kind': 'basket', 'place_color': 'none', 'sort_by': 'none'}`
- 'Separa las bolas de los cubos': want `{'intent': 'sort', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'kind'}` got `{'intent': 'not_doable', 'place_kind': 'basket', 'place_color': 'none', 'sort_by': 'none'}`
- 'No toques nada': want `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'stop', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'Cuéntame un chiste': want `{'intent': 'off_topic', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'greeting', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'Guarda todo en el cesto azul': want `{'intent': 'clear', 'place_kind': 'basket', 'place_color': 'blue', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'basket', 'place_color': 'blue', 'sort_by': 'none'}`
- '¿Puedes coger tres manzanas verdes?': want `{'intent': 'pick', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'green', 'sort_by': 'none'}`
- 'Cogen el objeto verde y dejaren el incesto rojo': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'color'}`
- 'coge la bola roja y mete la en el cesto verde.': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}`
- 'Coge la bola gorria': want `{'intent': 'pick', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}`
- 'Pick up the red ball and put it in the green basket': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'green', 'sort_by': 'none'}` got `{'intent': 'pick', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'none'}`
- 'put the blue cube in the yellow box': want `{'intent': 'pick_and_place', 'place_kind': 'box', 'place_color': 'yellow', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'box', 'place_color': 'blue', 'sort_by': 'none'}`
- "What's the weather like today?": want `{'intent': 'off_topic', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'question', 'place_kind': 'none', 'place_color': 'green', 'sort_by': 'none'}`
- 'Grab the green ball and toss it in the crate': want `{'intent': 'pick_and_place', 'place_kind': 'box', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'other', 'place_color': 'green', 'sort_by': 'none'}`
- 'Take the red sphere and place it in the basket': want `{'intent': 'pick_and_place', 'place_kind': 'basket', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'basket', 'place_color': 'red', 'sort_by': 'color'}`
- 'I need the yellow cube in the green box': want `{'intent': 'pick_and_place', 'place_kind': 'box', 'place_color': 'green', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'box', 'place_color': 'yellow', 'sort_by': 'none'}`
- 'Hand me the apple': want `{'intent': 'give', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'give', 'place_kind': 'other', 'place_color': 'pink', 'sort_by': 'none'}`
- 'Grab two blue balls and one red cube': want `{'intent': 'sequence', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'other', 'place_color': 'blue', 'sort_by': 'color'}`
- 'Sort everything by colour': want `{'intent': 'sort', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'color'}` got `{'intent': 'sort', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'Clear the table': want `{'intent': 'clear', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'clear', 'place_kind': 'table', 'place_color': 'none', 'sort_by': 'none'}`
- 'Stop!': want `{'intent': 'stop', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'stop', 'place_kind': 'other', 'place_color': 'none', 'sort_by': 'none'}`
- 'Stop moving right now': want `{'intent': 'stop', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'stop', 'place_kind': 'right', 'place_color': 'none', 'sort_by': 'none'}`
- 'Never mind': want `{'intent': 'cancel', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'not_doable', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}`
- 'The one on the left': want `{'intent': 'answer_value', 'place_kind': 'none', 'place_color': 'none', 'sort_by': 'none'}` got `{'intent': 'answer_no', 'place_kind': 'left', 'place_color': 'none', 'sort_by': 'none'}`
