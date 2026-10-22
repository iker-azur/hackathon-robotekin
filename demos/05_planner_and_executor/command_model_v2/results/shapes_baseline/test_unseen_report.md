# test_unseen.jsonl

1500 sentences, 135 ms per sentence

| | intent | items | place | sort_by | **command** |
| --- | --- | --- | --- | --- | --- |
| all | 97.3% | 55.9% | 91.1% | 100.0% | **51.6%** |
| en | 98.8% | 56.6% | 89.4% | 100.0% | **52.0%** |
| es | 97.0% | 63.0% | 89.4% | 100.0% | **55.4%** |
| eu | 96.2% | 48.0% | 94.6% | 100.0% | **47.4%** |

- stop caught: - (0/0)
- false action rate: 0.7% (2/270 sentences that must not move the robot)

| intent | recognised |
| --- | --- |
| answer_value | 98.3% (120) |
| give | 99.4% (180) |
| pick | 94.6% (240) |
| pick_and_place | 99.4% (540) |
| question | 100.0% (150) |
| sequence | 92.2% (270) |

## Errors (726)

- `es` 'pon la canica verde al contenedor azul': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "blue"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.745, "place_color": 0.863, "sort_by": 1.0}}`
- `en` 'first two green triangular-based prisms inside the tray, then the yellow bit in the hamper': wrong items, place  
  expected `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": "triangular-based prisms", "color": "green", "quantity": 2, "position": null, "place": {"kind": "tray", "color": null}}, {"kind": "any", "name": "bit", "color": "yellow", "quantity": 1, "position": null, "place": {"kind": "basket", "color": null}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": null, "color": "green", "quantity": 2, "position": null, "place": {"kind": "tray", "color": null}}, {"kind": "other", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": {"kind": "basket", "color": null}}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.966, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'pick the pink brick and move it inside the hamper': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "brick", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "brick", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.712, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'mmm, hiru adreilu urdin hartzea nahi dut': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "adreilu", "color": "blue", "quantity": 3, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "adreilu", "color": "blue", "quantity": 3, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Primero levanta la zanahoria verde y luego la esfera verde': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "other", "name": "zanahoria", "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "esfera", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "other", "name": null, "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "other", "name": "esfera", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.997, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'Stick the purple marbles into the hamper': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "marbles", "color": "purple", "quantity": "all", "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "marbles", "color": "purple", "quantity": "all", "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.726, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'adreilu horia kaxan sar ezazu': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "adreilu", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "adreilu", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'hay la esfera amarilla en la mesa': wrong place  
  expected `{"intent": "question", "items": [{"kind": "ball", "name": "esfera", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "question", "items": [], "place": {"kind": "table", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.998, "place_kind": 0.507, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'grab the blue brick': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "brick", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "brick", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'heldu oinarri triangeluarreko prisma zuria eta esfera zuria': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": "oinarri triangeluarreko prisma", "color": "white", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "esfera", "color": "white", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": null, "color": "white", "quantity": 1, "position": null, "place": null}, {"kind": "other", "name": null, "color": "white", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'um, hand me four pink triangular-based prisms, all the marbles and the black solid cylinder and put them to the right': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": "triangular-based prisms", "color": "pink", "quantity": 4, "position": null, "place": null}, {"kind": "ball", "name": "marbles", "color": null, "quantity": "all", "position": null, "place": null}, {"kind": "cylinder", "name": "solid cylinder", "color": "black", "quantity": 1, "position": null, "place": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": "triangular-based prisms", "color": "pink", "quantity": 4, "position": null, "place": null}, {"kind": "other", "name": null, "color": null, "quantity": "all", "position": null, "place": null}, {"kind": "cylinder", "name": null, "color": "black", "quantity": 1, "position": null, "place": null}], "place": {"kind": "right", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.62, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'first get the purple sphere and then the yellow one and put them inside the tray': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "sphere", "color": "purple", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": "one", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": "sphere", "color": "purple", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.97, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'eh, ¿puedes poner la canica amarilla a tu derecha?': wrong items, place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "canica", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "right", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "canica", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": {"kind": "left", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.704, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'kanika laranja hartu eta jar ezazu hozkailura': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "kanika", "color": "orange", "quantity": 1, "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "kanika", "color": "orange", "quantity": 1, "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.815, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Lleva en la canasta morada los prismas de base triangular naranjas': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "triangular_prism", "name": "prismas de base triangular", "color": "orange", "quantity": "all", "position": null, "place": null}], "place": {"kind": "basket", "color": "purple"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "triangular_prism", "name": null, "color": "orange", "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.998, "place_kind": 0.336, "place_color": 0.344, "sort_by": 1.0}}`
- `en` 'stick the yellow bricks into the yellow hamper': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bricks", "color": "yellow", "quantity": "all", "position": null, "place": null}], "place": {"kind": "basket", "color": "yellow"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "bricks", "color": "yellow", "quantity": "all", "position": null, "place": null}], "place": {"kind": "basket", "color": "yellow"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.857, "place_color": 0.799, "sort_by": 1.0}}`
- `en` 'bring me a green brick and leave it in the tray.': wrong items  
  expected `{"intent": "give", "items": [{"kind": "cube", "name": "brick", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "other", "name": "brick", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.996, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'leave into the hamper all the green triangular-based prisms': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "triangular_prism", "name": "triangular-based prisms", "color": "green", "quantity": "all", "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "triangular_prism", "name": "triangular-based prisms", "color": "green", "quantity": "all", "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.74, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Pásame una canica verde y déjalo en la bandeja': wrong items  
  expected `{"intent": "give", "items": [{"kind": "ball", "name": "canica", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "other", "name": "canica", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.995, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'mesedez, hartu esfera urdina eta sar ezazu kaxa gorrian': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "red"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "red"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.453, "place_color": 0.474, "sort_by": 1.0}}`
- `en` 'so, could you pick up the brick': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "brick", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "brick", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'bring me the yellow brick and the green sphere': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "cube", "name": "brick", "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "ball", "name": "sphere", "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "other", "name": "brick", "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "cube", "name": null, "color": "green", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.889, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'pon amarillo, rojo, amarillo y azul!': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'mesedez, hartu atzeko bloke zuria, ez, esfera berdea hartu eta eraman ezkerrean': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "left", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "left", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.997, "place_kind": 0.963, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'Utzi esfera berde guztiak mahai gainean, mesedez.': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "ball", "name": "esfera", "color": "green", "quantity": "all", "position": null, "place": null}], "place": {"kind": "table", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "esfera", "color": "green", "quantity": "all", "position": null, "place": null}], "place": {"kind": "table", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.998, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'could you place the red brick inside the crate': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "brick", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "brick", "color": "red", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.359, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'I want you to pick up the pink sphere': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "sphere", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "cube", "name": "sphere", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'quiero que cojas el prisma de base hexagonal rojo': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "hexagonal_prism", "name": "prisma de base hexagonal", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'tira, hartu adreilu urdina!': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "adreilu", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "adreilu", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'Alright, the brick': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "cube", "name": "brick", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "brick", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'necesito que cojas el ladrillo amarillo': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "ladrillo", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "ladrillo", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'levanta la esfera negra de delante y todas las zanahorias rojas': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "esfera", "color": "black", "quantity": 1, "position": "front", "place": null}, {"kind": "other", "name": "zanahorias", "color": "red", "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "other", "name": null, "color": "black", "quantity": 1, "position": "front", "place": null}, {"kind": "other", "name": "zanahorias", "color": "red", "quantity": "all", "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.996, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'Place the orange hexagonal-based prism in the tray and two orange hexagonal-based prisms into the crate!': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": "hexagonal-based prism", "color": "orange", "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}, {"kind": "hexagonal_prism", "name": "hexagonal-based prisms", "color": "orange", "quantity": 2, "position": null, "place": {"kind": "box", "color": null}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": null, "color": "orange", "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}, {"kind": "hexagonal_prism", "name": null, "color": "orange", "quantity": 2, "position": null, "place": {"kind": "box", "color": null}}], "place": {"kind": "box", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.976, "place_kind": 0.951, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Bueno, guarda dentro del contenedor el prisma de base hexagonal!': wrong place  
  expected `{"intent": "pick_and_place", "items": [{"kind": "hexagonal_prism", "name": "prisma de base hexagonal", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "hexagonal_prism", "name": null, "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "other", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.69, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'oinarri hexagonaleko prismak kaxa berdean sartu': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "hexagonal_prism", "name": "oinarri hexagonaleko prismak", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "hexagonal_prism", "name": "prismak", "color": null, "quantity": "all", "position": null, "place": null}], "place": {"kind": "box", "color": "green"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.327, "place_color": 0.89, "sort_by": 1.0}}`
- `eu` 'jarri laranja, urdina eta horia': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "any", "name": null, "color": "orange", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "any", "name": "laranja", "color": null, "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Robot, agarra un prisma de base hexagonal azul': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "hexagonal_prism", "name": "prisma de base hexagonal", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'Ados, adreilu beltza hartu eta sartu otzarara': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "adreilu", "color": "black", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "adreilu", "color": "black", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.788, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'la naranja': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "any", "name": null, "color": "orange", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "naranja", "color": "orange", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'la esfera azul': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "ball", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "esfera", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'recoge el bloque rosa, no, mejor el prisma de base hexagonal azul': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "hexagonal_prism", "name": "prisma de base hexagonal", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.578, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'take a brick and leave it inside the crate': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "brick", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "brick", "color": null, "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.995, "place_kind": 0.555, "place_color": 1.0, "sort_by": 1.0}}`
- `es` '¿me pones el trozo verde sobre la mesa naranja?': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "trozo", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "table", "color": "orange"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "trozo", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "table", "color": "orange"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.968, "place_color": 0.569, "sort_by": 1.0}}`
- `eu` 'hartu kanika, mesedez': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "kanika", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "kanika", "color": null, "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.998, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Recoge un ladrillo verde y mételo a la canasta': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "ladrillo", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "ladrillo", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.352, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'lift the blue marble': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "marble", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "marble", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'bring me all blue marbles and the pink marble and put them in the crate': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "marbles", "color": "blue", "quantity": "all", "position": null, "place": null}, {"kind": "ball", "name": "marble", "color": "pink", "quantity": 1, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "other", "name": "marbles", "color": "blue", "quantity": "all", "position": null, "place": null}, {"kind": "other", "name": null, "color": "pink", "quantity": 1, "position": null, "place": {"kind": "box", "color": null}}], "place": {"kind": "box", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.855, "place_kind": 0.832, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'jarri oinarri hexagonaleko prisma more bat otzarara eta bi oinarri triangeluarreko prisma more erretiluan': wrong place  
  expected `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": "oinarri hexagonaleko prisma", "color": "purple", "quantity": 1, "position": null, "place": {"kind": "basket", "color": null}}, {"kind": "triangular_prism", "name": "oinarri triangeluarreko prisma", "color": "purple", "quantity": 2, "position": null, "place": {"kind": "tray", "color": null}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": null, "color": "purple", "quantity": 1, "position": null, "place": {"kind": "basket", "color": null}}, {"kind": "triangular_prism", "name": null, "color": "purple", "quantity": 2, "position": null, "place": {"kind": "tray", "color": null}}], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.952, "place_color": 1.0, "sort_by": 0.999}}`
- `eu` 'hartu mesedez adreilu horia.': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "cube", "name": "adreilu", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "adreilu", "color": "yellow", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'Go ahead and grab the blue bit': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "any", "name": "bit", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "other", "name": "bit", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'Pásame las esferas blancas, cinco calcetines amarillos y tres prismas de base triangular azules': wrong intent, items  
  expected `{"intent": "sequence", "items": [{"kind": "ball", "name": "esferas", "color": "white", "quantity": "all", "position": null, "place": null}, {"kind": "other", "name": "calcetines", "color": "yellow", "quantity": 5, "position": null, "place": null}, {"kind": "triangular_prism", "name": "prismas de base triangular", "color": "blue", "quantity": 3, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "other", "name": null, "color": "white", "quantity": "all", "position": null, "place": null}, {"kind": "other", "name": null, "color": "yellow", "quantity": 5, "position": null, "place": null}, {"kind": "triangular_prism", "name": null, "color": "blue", "quantity": 3, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.571, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'Oinarri hexagonaleko prisma berdea nahi dut eta utzi erretilura': wrong items  
  expected `{"intent": "give", "items": [{"kind": "hexagonal_prism", "name": "oinarri hexagonaleko prisma", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "tray", "color": null}, "sort_by": "none"}`  
  got `{"intent": "give", "items": [{"kind": "hexagonal_prism", "name": "prisma", "color": "green", "quantity": 1, "position": null, "place": null}, {"kind": "triangular_prism", "name": "prisma", "color": "green", "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.984, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'esfera berdea otzarara': wrong items  
  expected `{"intent": "answer_value", "items": [{"kind": "ball", "name": "esfera", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none"}`  
  got `{"intent": "answer_value", "items": [{"kind": "other", "name": "esfera", "color": "green", "quantity": 1, "position": null, "place": null}], "place": {"kind": "basket", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.921, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'lehenengo heldu oinarri hexagonaleko prisma urdina eta gero hiru zati eta sartu mahai gainean.': wrong items  
  expected `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": "oinarri hexagonaleko prisma", "color": "blue", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": "zati", "color": null, "quantity": 3, "position": null, "place": null}], "place": {"kind": "table", "color": null}, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "hexagonal_prism", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}, {"kind": "triangular_prism", "name": "prisma", "color": "blue", "quantity": 1, "position": null, "place": null}, {"kind": "any", "name": null, "color": null, "quantity": 3, "position": null, "place": {"kind": "table", "color": null}}], "place": {"kind": "table", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.741, "place_kind": 0.998, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'Oinarri hexagonaleko prisma gorria hartzea nahi dut mesedez': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "hexagonal_prism", "name": "oinarri hexagonaleko prisma", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "triangular_prism", "name": "prisma", "color": "red", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'pick five yellow bricks and place them into the crate': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bricks", "color": "yellow", "quantity": 5, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "other", "name": "bricks", "color": "yellow", "quantity": 5, "position": null, "place": null}], "place": {"kind": "box", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 0.422, "place_color": 1.0, "sort_by": 1.0}}`
- `es` 'coges el prisma de base hexagonal azul': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "hexagonal_prism", "name": "prisma de base hexagonal", "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "any", "name": null, "color": "blue", "quantity": 1, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `eu` 'hartu bi esfera': wrong items  
  expected `{"intent": "pick", "items": [{"kind": "ball", "name": "esfera", "color": null, "quantity": 2, "position": null, "place": null}], "place": null, "sort_by": "none"}`  
  got `{"intent": "pick", "items": [{"kind": "cube", "name": "esfera", "color": null, "quantity": 2, "position": null, "place": null}], "place": null, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 1.0, "place_kind": 1.0, "place_color": 1.0, "sort_by": 1.0}}`
- `en` 'um, drop four green bits on the black table': wrong items  
  expected `{"intent": "pick_and_place", "items": [{"kind": "any", "name": "bits", "color": "green", "quantity": 4, "position": null, "place": null}], "place": {"kind": "table", "color": "black"}, "sort_by": "none"}`  
  got `{"intent": "pick_and_place", "items": [{"kind": "cube", "name": "bits", "color": "green", "quantity": 4, "position": null, "place": null}], "place": {"kind": "table", "color": "black"}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.999, "place_kind": 0.98, "place_color": 0.997, "sort_by": 1.0}}`
- `en` 'put two red triangular-based prisms in the crate and the stapler inside the tray, thanks': wrong items, place  
  expected `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": "triangular-based prisms", "color": "red", "quantity": 2, "position": null, "place": {"kind": "box", "color": null}}, {"kind": "other", "name": "stapler", "color": null, "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}], "place": null, "sort_by": "none"}`  
  got `{"intent": "sequence", "items": [{"kind": "triangular_prism", "name": null, "color": "red", "quantity": 2, "position": null, "place": {"kind": "box", "color": null}}, {"kind": "other", "name": null, "color": null, "quantity": 1, "position": null, "place": {"kind": "tray", "color": null}}], "place": {"kind": "tray", "color": null}, "sort_by": "none", "sort_filter": [], "confidences": {"intent": 0.963, "place_kind": 0.992, "place_color": 1.0, "sort_by": 1.0}}`
