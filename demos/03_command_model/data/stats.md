# Dataset statistics

- training sentences: 30000 (19397 different)
- training item phrases: 4266
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

- `en` **pick**: Grab the blue battery, thank you!  
  `{"intent": "pick", "items": [{"kind": "other", "name": "battery", "color": "blue", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **stop**: alto.  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **not_doable**: No muevas nada  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `es` **stop**: Para ya  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `eu` **clear**: jaso dena eta sartu erretilura  
  `{"intent": "clear", "items": [], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `eu` **give**: ekarridazu kubo berdea  
  `{"intent": "give", "items": [{"kind": "cube", "name": "kubo", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **pick_and_place**: Okay, set a black one on the orange table, please.  
  `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "one", "color": "black", "quantity": 1, "position": null}], "place": {"kind": "table", "color": "orange"}, "sort_by": "none"}`
- `eu` **repeat**: egin berriro.  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_value**: pilota zuria  
  `{"intent": "answer_value", "items": [{"kind": "ball", "name": "pilota", "color": "white", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **pick_and_place**: grab the red coin and stick it on top of the table  
  `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "coin", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "table", "color": null}, "sort_by": "none"}`
- `es` **cancel**: no importa  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `es` **pick_and_place**: ¿me pones todas las manzanas a la caja?  
  `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "manzanas", "color": null, "quantity": "all", "position": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`
- `eu` **give**: Objektu urdina nahi dut.  
  `{"intent": "give", "items": [{"kind": "any", "name": "objektu", "color": "blue", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **pick_and_place**: pilla la pelota roja y guárdala a la bandeja  
  `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "pelota", "color": "red", "quantity": 1, "position": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`
- `es` **not_doable**: mi hermano cogió la roja!  
  `{"intent": "not_doable", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **off_topic**: tell me a joke  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `en` **not_doable**: Don't move the pink ball  
  `{"intent": "not_doable", "items": [{"kind": "ball", "name": "ball", "color": "pink", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **off_topic**: ¿quién te ha programado?  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **pick**: Venga, coge el dado rojo  
  `{"intent": "pick", "items": [{"kind": "cube", "name": "dado", "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **sequence**: me traes la pelota blanca y la bola más cercana? y mételos en la caja morada  
  `{"intent": "sequence", "items": [{"kind": "ball", "name": "pelota", "color": "white", "quantity": 1, "position": null}, {"kind": "ball", "name": "bola", "color": null, "quantity": 1, "position": "nearest"}], "place": {"kind": "box", "color": "purple"}, "sort_by": "none"}`
- `eu` **pick**: Edalontzi denak har dezakezu?  
  `{"intent": "pick", "items": [{"kind": "other", "name": "edalontzi", "color": null, "quantity": "all", "position": null}], "place": null, "sort_by": "none"}`
- `en` **sequence**: pink, orange, orange, please  
  `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "pink", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "orange", "quantity": 1, "position": null}, {"kind": "any", "name": null, "color": "orange", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **give**: I want the green ball  
  `{"intent": "give", "items": [{"kind": "ball", "name": "ball", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **off_topic**: alright, do you like pizza?  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **off_topic**: venga, ¿dónde estamos?  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `en` **not_doable**: pick up the cat  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `en` **sequence**: fetch the black thing and the toy in front  
  `{"intent": "sequence", "items": [{"kind": "any", "name": "thing", "color": "black", "quantity": 1, "position": null}, {"kind": "other", "name": "toy", "color": null, "quantity": 1, "position": "front"}], "place": null, "sort_by": "none"}`
- `eu` **pick_and_place**: Ea, jarri kubo urdina saski urdinera  
  `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "kubo", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "basket", "color": "blue"}, "sort_by": "none"}`
- `eu` **not_doable**: har ezazu mahaia  
  `{"intent": "not_doable", "items": [], "place": null, "sort_by": "none"}`
- `es` **greeting**: hola, ¿como estas?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `eu` **greeting**: egun on  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_no**: wrong  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **clear**: robot, txukundu mahaia  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `eu` **not_doable**: jaurti pilota horia.  
  `{"intent": "not_doable", "items": [{"kind": "ball", "name": "pilota", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` **greeting**: pozten naiz zu ezagutzeaz  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `eu` **pick_and_place**: txanpona hartu eta sartu eskuinera  
  `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "txanpona", "color": null, "quantity": 1, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`
- `en` **repeat**: again  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `eu` **off_topic**: Jarri musika  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **sort**: agrupa los objetos por color  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "color"}`
- `eu` **cancel**: ez, utzi  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_value**: begira, pilota zuria  
  `{"intent": "answer_value", "items": [{"kind": "ball", "name": "pilota", "color": "white", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **question**: estás funcionando  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `es` **greeting**: hola, ¿cómo estás?  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_no**: okerra  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_value**: the yellow cable  
  `{"intent": "answer_value", "items": [{"kind": "other", "name": "cable", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **question**: what colour is the ball?  
  `{"intent": "question", "items": [{"kind": "ball", "name": "ball", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **answer_value**: la figura verde  
  `{"intent": "answer_value", "items": [{"kind": "any", "name": "figura", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` **sequence**: Lehenengo jaso kubo urdina eta gero lau sagar eta sartu eskuinaldean  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "kubo", "color": "blue", "quantity": 1, "position": null}, {"kind": "other", "name": "sagar", "color": null, "quantity": 4, "position": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`
- `eu` **repeat**: Egin ezazu berriro!  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `es` **sort**: a ver, agrupa todas las piezas por color.  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "color"}`
- `en` **sort**: sort everything by colour  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "color"}`
- `en` **pick**: Pick the orange one.  
  `{"intent": "pick", "items": [{"kind": "any", "name": "one", "color": "orange", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **clear**: Pick everything up and put it into the basket, please.  
  `{"intent": "clear", "items": [], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`
- `en` **question**: right, where is the box  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `eu` **off_topic**: Nekatuta nago  
  `{"intent": "off_topic", "items": [], "place": null, "sort_by": "none"}`
- `es` **sequence**: primero recoge un cubo negro y luego la manzana azul y mételos en la izquierda  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "cubo", "color": "black", "quantity": 1, "position": null}, {"kind": "other", "name": "manzana", "color": "blue", "quantity": 1, "position": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`
- `en` **clear**: clean up the table  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `en` **stop**: stop it  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `en` **stop**: stop now!  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `en` **sort**: sort the pieces by type  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "kind"}`
- `eu` **sequence**: hartu mesedez kubo guztiak eta bi pilota  
  `{"intent": "sequence", "items": [{"kind": "cube", "name": "kubo", "color": null, "quantity": "all", "position": null}, {"kind": "ball", "name": "pilota", "color": null, "quantity": 2, "position": null}], "place": null, "sort_by": "none"}`
- `eu` **question**: Zenbat pieza daude  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `en` **cancel**: scratch that  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `en` **greeting**: Good night  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_value**: on the blue table  
  `{"intent": "answer_value", "items": [], "place": {"kind": "table", "color": "blue"}, "sort_by": "none"}`
- `eu` **sort**: bereizi objektuak koloreka  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "color"}`
- `eu` **sort**: banandu piezak kolorearen arabera  
  `{"intent": "sort", "items": [], "place": null, "sort_by": "color"}`
- `eu` **stop**: geldirik  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **pick**: recoge dos piezas  
  `{"intent": "pick", "items": [{"kind": "any", "name": "piezas", "color": null, "quantity": 2, "position": null}], "place": null, "sort_by": "none"}`
- `eu` **pick**: hartuko zenuke pieza berdea?  
  `{"intent": "pick", "items": [{"kind": "any", "name": "pieza", "color": "green", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **greeting**: bye  
  `{"intent": "greeting", "items": [], "place": null, "sort_by": "none"}`
- `es` **clear**: despeja la mesa  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_no**: ez  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **question**: ikus dezakezu giltza zuria  
  `{"intent": "question", "items": [{"kind": "other", "name": "giltza", "color": "white", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **question**: ¿qué tienes en la pinza?  
  `{"intent": "question", "items": [], "place": null, "sort_by": "none"}`
- `eu` **answer_yes**: hori bera  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `en` **cancel**: Leave it  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `en` **answer_yes**: yeah!  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **clear**: venga, limpia la mesa  
  `{"intent": "clear", "items": [], "place": null, "sort_by": "none"}`
- `es` **give**: ¿me pasas la roja?  
  `{"intent": "give", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `es` **answer_value**: la llave  
  `{"intent": "answer_value", "items": [{"kind": "other", "name": "llave", "color": null, "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **answer_no**: nope!  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `eu` **stop**: ez, gelditu  
  `{"intent": "stop", "items": [], "place": null, "sort_by": "none"}`
- `es` **repeat**: otra vez  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `es` **cancel**: no importa  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_yes**: venga  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `eu` **cancel**: ahaztu hori  
  `{"intent": "cancel", "items": [], "place": null, "sort_by": "none"}`
- `en` **give**: Hand me the orange ball  
  `{"intent": "give", "items": [{"kind": "ball", "name": "ball", "color": "orange", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `eu` **answer_yes**: Hori bera  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **give**: ¿me das el dado amarillo?  
  `{"intent": "give", "items": [{"kind": "cube", "name": "dado", "color": "yellow", "quantity": 1, "position": null}], "place": null, "sort_by": "none"}`
- `en` **answer_yes**: that one.  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `es` **repeat**: hazlo de nuevo.  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_no**: No  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_no**: esa no  
  `{"intent": "answer_no", "items": [], "place": null, "sort_by": "none"}`
- `es` **answer_yes**: de acuerdo.  
  `{"intent": "answer_yes", "items": [], "place": null, "sort_by": "none"}`
- `en` **repeat**: again, please  
  `{"intent": "repeat", "items": [], "place": null, "sort_by": "none"}`
