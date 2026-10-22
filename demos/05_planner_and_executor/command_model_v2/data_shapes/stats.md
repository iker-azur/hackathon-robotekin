# Dataset statistics

- training sentences: 30000 (19623 different)
- training item phrases: 3805
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

- `en` **pick_and_place**: Lift the blue ball and place it on the table  
  `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "ball", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "table", "color": null}, "sort_by": "none"}`
- `en` **pick**: fetch the yellow ball  
  `{"intent": "pick", "items": [{"kind": "ball", "name": "ball", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **not_doable**: throw the orange cube  
  `{"intent": "not_doable", "items": [{"kind": "cube", "name": "cube", "color": "orange", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **off_topic**: zenbat da bi gehi bi  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **pick_and_place**: mete el prisma hexagonal rojo en el cesto  
  `{"intent": "pick_and_place", "items": [{"kind": "hexagonal_prism", "name": "prisma hexagonal", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `en` **answer_no**: Not that one  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `en` **off_topic**: do you love me  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `eu` **sequence**: Jarri kubo berdea erretiluan eta bola berdea saskian  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "kubo", "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}, {"kind": "ball", "name": "bola", "color": "green", "quantity": 1, "position": null, "place": {"kind": "basket", "color": null}}], "place": null, "sort_by": "none"}`
- `en` **cancel**: never mind  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **repeat**: egin berriro.  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `en` **pick_and_place**: leave inside the drawer the white triangular prism  
  `{"intent": "pick_and_place", "items": [{"kind": "triangular_prism", "name": "triangular prism", "color": "white", "quantity": 1, "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`
- `en` **pick**: pick up the object at the back, please  
  `{"intent": "pick", "items": [{"kind": "any", "name": "object", "color": null, "quantity": 1, "position": "back", "place": null}], "place": null, "sort_by": "none"}`
- `eu` **give**: eman prisma triangeluar urdina eta utzi erretiluan  
  `{"intent": "give", "items": [{"kind": "triangular_prism", "name": "prisma triangeluar", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `es` **clear**: quita las pelotas azules del fondo de la mesa y ponlos en el cesto  
  `{"intent": "clear", "items": [{"kind": "ball", "name": "pelotas", "color": "blue", "quantity": "all", "position": "back", "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `en` **not_doable**: pick up the car  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `eu` **pick**: mesedez, hartu gauza arrosa.  
  `{"intent": "pick", "items": [{"kind": "any", "name": "gauza", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **question**: ¿ves la pelota roja?  
  `{"intent": "question", "items": [{"kind": "ball", "name": "pelota", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **greeting**: hola, ¿qué tal?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_yes**: affirmative  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **greeting**: Nos vemos  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **clear**: clear everything  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `es` **not_doable**: Ayer cogí el cilindro amarillo  
  `{"intent": "not_doable", "items": [{"kind": "cylinder", "name": "cilindro", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **stop**: para, para.  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **sequence**: primero toma un prisma triangular y luego un cilindro verde  
  `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": "prisma triangular", "color": null, "quantity": 1, "position": null, "place": null}, {"kind": "cylinder", "name": "cilindro", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **sequence**: take the yellow hexagonal prism and the blue cylinder and put them inside the red box  
  `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": "hexagonal prism", "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "cylinder", "name": "cylinder", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`
- `es` **pick_and_place**: Mete a la caja azul el prisma hexagonal naranja.  
  `{"intent": "pick_and_place", "items": [{"kind": "hexagonal_prism", "name": "prisma hexagonal", "color": "orange", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`
- `en` **off_topic**: do you like pizza  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **repeat**: Repite  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **pick_and_place**: Sar ezazu hiru dado berde saskira  
  `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "dado", "color": "green", "quantity": 3, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `eu` **off_topic**: maite nauzu?  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_value**: eskuineko prisma triangeluar horia.  
  `{"intent": "answer_value", "items": [{"kind": "triangular_prism", "name": "prisma triangeluar", "color": "yellow", "quantity": 1, "position": "right", "place": null}], "place": null, "sort_by": "none"}`
- `eu` **sort**: bereizi piezak koloreka.  
  `{"intent": "sort", "items": [{"kind": "any", "name": "piezak", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "color"}`
- `en` **question**: what are you doing  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `es` **question**: qué puedes hacer  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `en` **clear**: pick everything up  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `eu` **pick_and_place**: heldu prisma triangeluar horia, ez, hobeto bola berde guztiak lurrera eraman!  
  `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "bola", "color": "green", "quantity": "all", "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`
- `en` **give**: Robot, I want the triangular prism and leave it into the tray, thanks  
  `{"intent": "give", "items": [{"kind": "triangular_prism", "name": "triangular prism", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `en` **give**: hey robot, get me all pears  
  `{"intent": "give", "items": [{"kind": "other", "name": "pears", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **repeat**: saiatu berriz!  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_value**: el prisma hexagonal  
  `{"intent": "answer_value", "items": [{"kind": "hexagonal_prism", "name": "prisma hexagonal", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **cancel**: nada, déjalo  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `es` **pick**: recoge el prisma hexagonal rojo.  
  `{"intent": "pick", "items": [{"kind": "hexagonal_prism", "name": "prisma hexagonal", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **greeting**: gabon  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `es` **clear**: quita los rojos de la mesa y ponlos en la bandeja por favor  
  `{"intent": "clear", "items": [{"kind": "any", "name": null, "color": "red", "quantity": "all", "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `es` **sort**: mmm, separa los cubos de las bolas  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "kind"}`
- `en` **stop**: freeze  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **off_topic**: Qué calor hace!  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `eu` **not_doable**: hartu Jon  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_yes**: sí, eso  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_value**: all blue cylinders.  
  `{"intent": "answer_value", "items": [{"kind": "cylinder", "name": "cylinders", "color": "blue", "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **sequence**: Alright, first get the green ball on the left and then three strawberries!  
  `{"intent": "sequence", "items": [{"kind": "ball", "name": "ball", "color": "green", "quantity": 1, "position": "left", "place": null}, {"kind": "other", "name": "strawberries", "color": null, "quantity": 3, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `eu` **sequence**: mesedez, hartu prisma hexagonalak eta dado horia eta sartu kutxan  
  `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": "prisma hexagonalak", "color": null, "quantity": "all", "position": null, "place": null}, {"kind": "cube", "name": "dado", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `es` **cancel**: No importa  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `en` **greeting**: Hey  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **stop**: don't move!  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **not_doable**: Recoge el sol  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_value**: mmm, la esponja dentro del cesto  
  `{"intent": "answer_value", "items": [{"kind": "other", "name": "esponja", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `es` **stop**: para ya.  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `en` **question**: do you see the black cylinder?  
  `{"intent": "question", "items": [{"kind": "cylinder", "name": "cylinder", "color": "black", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **pick**: bueno, coge un amarillo  
  `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **sort**: clasifica los cilindros por forma y ponlas dentro de la caja  
  `{"intent": "sort", "items": [{"kind": "cylinder", "name": "cilindros", "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "kind"}`
- `es` **repeat**: vuelve a intentarlo  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `en` **cancel**: Leave it  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `es` **give**: acércame el cilindro y déjalo en la bandeja  
  `{"intent": "give", "items": [{"kind": "cylinder", "name": "cilindro", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `eu` **stop**: Ez mugitu.  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **give**: ¿me pasas el prisma triangular morado?  
  `{"intent": "give", "items": [{"kind": "triangular_prism", "name": "prisma triangular", "color": "purple", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **greeting**: Hello, how are you?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `eu` **clear**: Ados, gorde dena saskira!  
  `{"intent": "clear", "items": [], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `eu` **cancel**: ahaztu esan dizudana  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **pick**: mesedez, hartu pieza gorria... ez, itxaron, bola arrosa!  
  `{"intent": "pick", "items": [{"kind": "ball", "name": "bola", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **answer_no**: Wrong  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_value**: atzeko edalontzia  
  `{"intent": "answer_value", "items": [{"kind": "other", "name": "edalontzia", "color": null, "quantity": 1, "position": "back", "place": null}], "place": null, "sort_by": "none"}`
- `eu` **not_doable**: Ez egin ezer  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `eu` **clear**: Jaso dena eta sartu ezkerrera  
  `{"intent": "clear", "items": [], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `es` **off_topic**: Gracias  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `en` **sort**: sort the cubes by color  
  `{"intent": "sort", "items": [{"kind": "cube", "name": "cubes", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "color"}`
- `es` **answer_no**: para nada  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **stop**: nahikoa  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `eu` **sort**: bereizi kuboak eta bolak  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "kind"}`
- `en` **repeat**: One more time!  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **question**: zer ikusten duzu  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `eu` **question**: ikusten duzu prisma triangeluar berdea  
  `{"intent": "question", "items": [{"kind": "triangular_prism", "name": "prisma triangeluar", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `en` **repeat**: Do it again  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_value**: The red cylinder at the back  
  `{"intent": "answer_value", "items": [{"kind": "cylinder", "name": "cylinder", "color": "red", "quantity": 1, "position": "back", "place": null}], "place": null, "sort_by": "none"}`
- `en` **sort**: sort the clips by type  
  `{"intent": "sort", "items": [{"kind": "other", "name": "clips", "color": null, "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "kind"}`
- `eu` **greeting**: Aupa.  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `eu` **give**: Eman kubo zuri bat  
  `{"intent": "give", "items": [{"kind": "cube", "name": "kubo", "color": "white", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`
- `es` **sequence**: Primero levanta el cubo amarillo y luego todos los cilindros y mételos a la bandeja.  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "cubo", "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "cylinder", "name": "cilindros", "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `es` **answer_yes**: exacto  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `eu` **cancel**: ez du axola  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_no**: hori ez  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_yes**: sounds good  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_no**: negativo  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_no**: Ez.  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_yes**: Ondo  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_yes**: bai, mesedez  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
