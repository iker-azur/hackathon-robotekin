# Dataset statistics

- training sentences: 30000 (19815 different)
- training item phrases: 4428
- unseen-word test sentences: 1500

| intent | es | en | eu |
| --- | --- | --- | --- |
| pick_and_place | 1800 | 1800 | 1800 |
| pick | 800 | 800 | 800 |
| give | 600 | 600 | 600 |
| sort | 500 | 500 | 500 |
| clear | 400 | 400 | 400 |
| sequence | 900 | 900 | 900 |
| stop | 400 | 400 | 400 |
| repeat | 250 | 250 | 250 |
| cancel | 250 | 250 | 250 |
| answer_yes | 250 | 250 | 250 |
| answer_no | 250 | 250 | 250 |
| answer_value | 400 | 400 | 400 |
| question | 500 | 500 | 500 |
| greeting | 600 | 600 | 600 |
| off_topic | 900 | 900 | 900 |
| not_doable | 1200 | 1200 | 1200 |

## Examples

- `eu` **answer_yes**: Ondo  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **sequence**: Primero agarra un dado blanco y luego la bola naranja, gracias  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "dado", "color": "white", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "bola", "color": "orange", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **pick_and_place**: take the yellow screwdriver... no, wait, fetch all green balls and place them on the right  
  `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "balls", "color": "green", "quantity": "all", "position": null, "place": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`
- `es` **off_topic**: Oye, llama a mi madre  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_yes**: that's right  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `en` **pick**: Fetch two ones  
  `{"intent": "pick", "items": [{"kind": "any", "name": "ones", "color": null, "quantity": 2, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **clear**: garbitu mahaia  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `eu` **give**: bola berdea behar dut eta utzi erretilura  
  `{"intent": "give", "items": [{"kind": "ball", "name": "bola", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `en` **not_doable**: get the car.  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `eu` **off_topic**: pizza gustatzen zaizu?  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `en` **off_topic**: where are you from?  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **question**: Vale, ¿qué estás haciendo?  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `es` **repeat**: inténtalo de nuevo  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_yes**: sí, eso  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `eu` **stop**: nahikoa  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **stop**: quieto!  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_value**: bola beltza mahai gainean  
  `{"intent": "answer_value", "items": [{"kind": "ball", "name": "bola", "color": "black", "quantity": 1, "position": null, "place": null}], "place": {"kind": "table", "color": null}, "sort_by": "none"}`
- `en` **not_doable**: Fetch the computer  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_no**: Incorrect  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `es` **greeting**: Hola, ¿cómo estás  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `es` **pick_and_place**: deja la roja dentro del cesto.  
  `{"intent": "pick_and_place", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `es` **pick_and_place**: levanta tres cubos rosas y mételos en la caja.  
  `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "cubos", "color": "pink", "quantity": 3, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `es` **greeting**: Buenas noches  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **off_topic**: Turn off the lights  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `eu` **sequence**: mmm, jaso errotulagailua eta sagarra  
  `{"intent": "sequence", "items": [{"kind": "other", "name": "errotulagailua", "color": null, "quantity": 1, "position": null, "place": null}, {"kind": "other", "name": "sagarra", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **pick_and_place**: utzi errotulagailu denak lurrean  
  `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "errotulagailu", "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`
- `en` **greeting**: alright, good evening  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **give**: I want all blue ones and leave it into the tray  
  `{"intent": "give", "items": [{"kind": "any", "name": "ones", "color": "blue", "quantity": "all", "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `eu` **pick_and_place**: har ezazu arkatza eta eraman ezkerraldean  
  `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "arkatza", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `en` **give**: Give me the block and leave it in the tray!  
  `{"intent": "give", "items": [{"kind": "cube", "name": "block", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `en` **pick_and_place**: alright, stick three blue blocks in the basket  
  `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "blocks", "color": "blue", "quantity": 3, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `es` **pick**: Vale, coge todas las bolas rojas  
  `{"intent": "pick", "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **stop**: Stop it  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `eu` **not_doable**: nire anaiak urdina hartu zuen  
  `{"intent": "not_doable", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **sequence**: lehenengo jaso kubo arrosa eta gero pieza eta gero pieza morea eta sartu mahai laranjaren gainean.  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "kubo", "color": "pink", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": "pieza", "color": null, "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": "pieza", "color": "purple", "quantity": 1, "position": null, "place": null}], "place": {"kind": "table", "color": "orange"}, "sort_by": "none"}`
- `en` **sort**: Sort the items by colour  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "color"}`
- `en` **answer_yes**: yes, please  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **not_doable**: antes había la llave roja en la mesa.  
  `{"intent": "not_doable", "items": [{"kind": "other", "name": "llave", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **answer_value**: dentro de la caja verde  
  `{"intent": "answer_value", "items": [], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`
- `eu` **repeat**: egin ezazu berriro  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_no**: ezetz  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_yes**: Hori da.  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_value**: the green ball  
  `{"intent": "answer_value", "items": [{"kind": "ball", "name": "ball", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **off_topic**: Zein egun polita  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **sequence**: quiero verde, amarillo, azul y amarillo  
  `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **repeat**: Hey robot, repeat  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_value**: la naranja  
  `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": "orange", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **not_doable**: Ez ukitu pilota urdina.  
  `{"intent": "not_doable", "items": [{"kind": "ball", "name": "pilota", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **pick**: Toma todos los libros!  
  `{"intent": "pick", "items": [{"kind": "other", "name": "libros", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **pick**: ea, hartu aurreko eraztun horia  
  `{"intent": "pick", "items": [{"kind": "other", "name": "eraztun", "color": "yellow", "quantity": 1, "position": "front", "place": null}], "place": null, "sort_by": "none"}`
- `es` **off_topic**: oye, el examen fue difícil  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `en` **pick**: Fetch the blue ball  
  `{"intent": "pick", "items": [{"kind": "ball", "name": "ball", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **give**: ekarridazu giltza beltzak eta utzi erretiluan  
  `{"intent": "give", "items": [{"kind": "other", "name": "giltza", "color": "black", "quantity": "all", "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `eu` **repeat**: berriro  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `en` **greeting**: how's it going?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `es` **sort**: Ordena las cosas azules dentro de la cesta azul y los martillos rojos en la bandeja roja!  
  `{"intent": "sort", "items": [{"kind": "any", "name": "cosas", "color": "blue", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "blue"}}, {"kind": "other", "name": "martillos", "color": "red", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "red"}}], "place": null, "sort_by": "color"}`
- `eu` **sort**: taldekatu bolak kolorez.  
  `{"intent": "sort", "items": [{"kind": "ball", "name": "bolak", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "color"}`
- `es` **clear**: quita los objetos de la mesa y ponlos dentro de la bandeja.  
  `{"intent": "clear", "items": [{"kind": "any", "name": "objetos", "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `es` **repeat**: Vuelve a hacerlo!  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **sort**: Sartu pila gorriak erretilu gorrian eta pilota urdinak saski urdinera.  
  `{"intent": "sort", "items": [{"kind": "other", "name": "pila", "color": "red", "quantity": "all", "position": null, "place": {"kind": "tray", "color": "red"}}, {"kind": "ball", "name": "pilota", "color": "blue", "quantity": "all", "position": null, "place": {"kind": "basket", "color": "blue"}}], "place": null, "sort_by": "color"}`
- `en` **answer_no**: negative!  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **pick**: hartuko zenuke kubo arrosa?  
  `{"intent": "pick", "items": [{"kind": "cube", "name": "kubo", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **question**: what can you do?  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `es` **clear**: ordena la mesa  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `es` **not_doable**: Vale, no toques nada  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `en` **clear**: Move everything into the tray!  
  `{"intent": "clear", "items": [], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `es` **stop**: Para, para  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `eu` **stop**: gelditu  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **question**: ¿dónde está la caja?  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `eu` **greeting**: Zer moduz zaude?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_value**: four cables  
  `{"intent": "answer_value", "items": [{"kind": "other", "name": "cables", "color": null, "quantity": 4, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **clear**: clean up the table  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `es` **cancel**: Mejor no  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **greeting**: kaixo, zer berri?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `es` **give**: pásame la tuerca, porfa y déjalo en la bandeja  
  `{"intent": "give", "items": [{"kind": "other", "name": "tuerca", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `en` **cancel**: cancel  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `en` **stop**: stop, stop  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_value**: Kikara!  
  `{"intent": "answer_value", "items": [{"kind": "other", "name": "kikara", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **sequence**: place the apple inside the box and the purple eraser into the box, thank you  
  `{"intent": "sequence", "items": [{"kind": "other", "name": "apple", "color": null, "quantity": 1, "position": null, "place": {"kind": "box", "color": null}}, {"kind": "other", "name": "eraser", "color": "purple", "quantity": 1, "position": null, "place": {"kind": "box", "color": null}}], "place": null, "sort_by": "none"}`
- `es` **give**: A ver, ¿me pasas tres cubos amarillos?  
  `{"intent": "give", "items": [{"kind": "cube", "name": "cubos", "color": "yellow", "quantity": 3, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **sort**: agrupa todo por forma, porfa  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "kind"}`
- `en` **sort**: separate the cubes from the balls and put them inside the bin  
  `{"intent": "sort", "items": [], "place": {"kind": "basket", "color": null}, "sort_by": "kind"}`
- `en` **sequence**: bring me a green ball, the blue balls and the yellow one and put them inside the box, please  
  `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "balls", "color": "blue", "quantity": "all", "position": null, "place": null}, {"kind": "any", "name": "one", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `es` **cancel**: cancela la orden  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `en` **repeat**: repeat that.  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **question**: non dago kutxa?  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `en` **cancel**: scratch that  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **cancel**: ahaztu hori!  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **clear**: garbitu mahaia.  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_no**: Hori ez  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **question**: zenbat pieza daude?  
  `{"intent": "question", "items": [{"kind": "any", "name": "pieza", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **answer_yes**: de acuerdo  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `en` **question**: what objects are there?  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_no**: No es eso.  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **cancel**: ahaztu  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_no**: no es eso  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
