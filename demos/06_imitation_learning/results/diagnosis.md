# Diagnóstico: por qué las políticas ACT con cámara no completaban la tarea

Fecha: 2026-10-19. Entorno `hackathon-robotekin`, Windows 11, Isaac Sim 5.1, RTX 5060 Ti.
Parte del commit `a3db19f` ("add phase and detector inputs to imitation demo").

Todo lo de este documento sale de una medición hecha sobre los artefactos que hay en
disco (parquet, vídeo, checkpoints) o en el simulador, y cada apartado dice cómo se
volvió a medir. La versión anterior de este archivo contenía conclusiones que estas
mediciones contradicen; están marcadas como superadas al final.

## Resumen

1. **El modelo no está roto ni mal entrenado.** Sobre las demostraciones de entrenamiento
   reproduce la acción del experto con errores de 1–5 mm en `go above`, `go down` y
   `close`, también en `go down` (z predicha 0,141 m frente a 0,144 m del experto).
   La pérdida de entrenamiento era una buena medida *de eso*.
2. **La entrada llega bien al modelo.** Reproduciendo el experto en el simulador, las
   imágenes y estados que ve el modelo coinciden con los del dataset y sus predicciones
   difieren milímetros.
3. **El fallo está en el lazo cerrado y es un atajo de estado.** La política copia la pose
   del propio brazo: dando a la misma imagen el estado articular de *otro* episodio, la
   y predicha se mueve 20–45 mm cuando el experto no se movería nada. Si el brazo se
   desvía, la política lo sigue y no hay nada que lo devuelva al cubo.
4. **Medir mal lo ocultaba.** La evaluación reutilizaba las posiciones de las
   demostraciones, no reiniciaba la cola de acciones de ACT, juzgaba solo la posición
   final del cubo y el primer fotograma de cada episodio mostraba la escena del
   episodio anterior.
5. **El agarre físico funciona**, pero con una ventana estrecha y descentrada: el
   experto original apuntaba a unos 5 mm del borde. El guion levanta y coloca el cubo **por
   fricción, sin sujeción cinemática**, en 10/10 posiciones fuera del entrenamiento, tanto con
   el punto de agarre original como con el desplazado; el desplazamiento solo da margen frente
   al error de percepción.
6. **La cámara se movía en cada episodio, y el demo creía tenerla congelada.**
   `il_env.set_cube_randomization` recorría `dataclasses.fields(cfg.events)`, pero LeIsaac
   cuelga sus términos de aleatorización como atributos normales: el bucle no encontraba
   ninguno y no congelaba nada. La pose de la cámara (±5 mm, ±0,05°) se redibujaba en cada
   `env.reset()`, así que la misma posición del cubo caía en píxeles distintos según el
   episodio. Con la cámara moviéndose, el centroide del cubo predice su posición en la mesa
   con 5 mm de error medio (14 mm de peor caso); congelada, con **0,5 mm (2,3 mm)**. Todos
   los datasets anteriores llevan ese temblor en sus imágenes.

## 1. El modelo reproduce lo que vio (`15_act_phase`, fase como entrada)

Predicción de la primera acción de cada bloque sobre 4 episodios de entrenamiento
(0, 7, 19, 33), un fotograma de cada 10, con los preprocesadores del propio checkpoint;
error medio absoluto en centímetros:

| fase | muestras | dx | dy | dz | z experto / z predicha (m) | signo de la pinza |
|---|---|---|---|---|---|---|
| go above cube | 40 | 0,44 | 0,18 | 0,29 | 0,207 / 0,205 | 100 % |
| go down | 40 | 0,40 | 0,21 | 0,34 | 0,144 / 0,141 | 100 % |
| close | 24 | 0,56 | 0,20 | 0,21 | 0,138 / 0,136 | 100 % |
| lift | 16 | 0,85 | 0,16 | 1,16 | 0,187 / 0,199 | 100 % |
| carry | 48 | 0,48 | 0,14 | 0,09 | 0,198 / 0,199 | 100 % |
| put down | 32 | 0,29 | 0,17 | 0,55 | 0,145 / 0,139 | 100 % |
| open | 16 | 0,18 | 0,16 | 0,20 | 0,138 / 0,136 | 100 % |
| retract | 40 | 0,33 | 0,20 | 0,22 | 0,192 / 0,194 | 100 % |

(La pérdida final 0,055 de los entrenamientos anteriores es esto: un buen ajuste. Lo que
no predice es el lazo cerrado.)

## 2. Lo que recibe el modelo en el simulador es lo que vio al entrenar

Se reprodujo el episodio 0 del experto en Isaac Sim y se compararon, fotograma a
fotograma (0, 1, 5, 50, 99, 110, 150), imagen y estado del simulador con los del dataset:

* diferencia media de imagen 3,4–4,5 sobre 255 (compresión AV1);
* diferencia de estado articular 0,000 (0,073° en el paso 110);
* predicción con entrada del simulador y con entrada del dataset: iguales a unos 3 mm.
  Por ejemplo, paso 110: experto (0,022; −0,215; 0,145) m, predicción con el dataset
  (0,024; −0,214; 0,142), con el simulador (0,027; −0,213; 0,142).

Quedan descartados: normalización, orden de canales, unidades de la acción, escala de la
imagen, preprocesado y postprocesado.

## 3. Traza del lazo cerrado (`15_act_phase`, episodio 0 de entrenamiento)

```
paso  10 [go above] xyz=(+0,053; −0,253; +0,208)    experto en el fotograma 10: (+0,022; −0,222; +0,208)
paso 100 [go above] xyz=(+0,048; −0,230; +0,199)    experto: (+0,022; −0,215; +0,198)
paso 110 [go down ] xyz=(+0,049; −0,229; +0,142)    experto: (+0,022; −0,215; +0,145)
paso 200 [go down ] xyz=(+0,060; −0,231; +0,136)    experto: (+0,022; −0,215; +0,138)
paso 260 [close   ] xyz=(+0,071; −0,234; +0,136)    cubo (0,319; −0,431), empujado 1,8 cm; error final 11,6 cm
```

La z **sí baja** al entrar en `go down` con la fase como entrada, así que "la z se queda
en la altura de aproximación" describe al checkpoint *sin* fase (`15_act_cube_detect`),
que no tiene forma de saber cuándo bajar. Con fase, el fallo es otro: x lleva 3–5 cm de
desvío y la pinza baja sobre el borde del cubo y lo empuja.

## 4. El atajo de estado

Sobre `15_act_phase`, fotograma 70 (el brazo ha llegado sobre el cubo) de los episodios
0, 7, 19 y 33. Se cambia una sola entrada y se mira cuánto se mueve la posición
predicha (mm) frente a cuánto debería moverse según el experto:

| entrada cambiada (dx; dy en mm) | ep 0 | ep 7 | ep 19 | ep 33 | el experto movería |
|---|---|---|---|---|---|
| estado articular de otro episodio, misma imagen | 8,9; **−43,3** | −8,2; **+45,5** | −13,2; **+35,7** | −12,9; **+20,5** | 0; 0 (el cubo es el mismo) |
| imagen de otro episodio, mismo estado | 34,8; −8,8 | −34,1; 10,9 | −20,8; 6,1 | −9,8; 2,6 | 39; −50 · −39; 50 · −31; 37 · −21; 21 |
| una articulación +3° (la mayor de las cinco) | ≤ 5 | ≤ 3 | ≤ 4 | ≤ 5 | — |

La y predicha sale sobre todo del estado articular y la x de las dos cosas. En las
demostraciones el brazo está siempre donde acaba de mandarlo el experto, así que
"repite la pose que ya tienes" ajusta la pérdida igual de bien que "ve dónde está el cubo"
y es más fácil de aprender. En el lazo cerrado, cualquier desvío del brazo se convierte en
una orden desviada igual, y de ahí en un desvío mayor.

Es el fenómeno que la literatura llama *causal confusion* (de Haan et al., 2019). No lo
arregla un bloque de acciones más largo ni más pasos de entrenamiento: lo arregla quitar
la pose del brazo de lo que la política puede leer.

## 5. Lo que ocultaba la evaluación

| hallazgo | cómo se midió | efecto |
|---|---|---|
| Las posiciones evaluadas eran las de entrenamiento | `sample_cube_spot(i)` en demo 13 y en demo 16 | "episodio 0" era la demostración 0, con cubo y yaw idénticos |
| `policy.reset()` no se llamaba entre episodios | lectura de `select_action` de LeRobot: la cola de acciones sobrevive | el episodio siguiente empezaba repitiendo el final del anterior (invisible con 800 pasos porque 800 es múltiplo de 20; visible con `--steps 260`) |
| El criterio era la posición final del cubo | un cubo empujado 10 cm acaba "en el sitio" | un empuje contaba como éxito |
| El informe se escribía siempre en `results/16_evaluation.md` | una ejecución de diagnóstico de esta sesión lo sobrescribió (restaurado con `git checkout`) | se perdía el registro completo; ahora hay un archivo por ejecución y no se sobrescribe |
| El primer fotograma de cada episodio es de la escena anterior | diferencia entre fotogramas 1 y 2: 5,5 en los episodios ≥ 1 frente a 3,2 en el 0; el detector sobre el fotograma 0 da un error de ajuste lineal de 19 mm de media (peor caso 91 mm) y sobre los fotogramas 2–30 de 6,2 mm (13 mm) | el simulador avanza la física con `render=False` en el reinicio, y la cámara entrega además el render *anterior* al pedido: hacen falta dos renders |
| La cámara va a 30 Hz con física a 60 Hz | diferencia exactamente 0 entre los fotogramas 0 y 1, 2 y 3, 4 y 5 | cada imagen del dataset está duplicada |

## 6. El agarre físico: sin sujeción cinemática, con una ventana estrecha

El experto fijaba el cubo a la pinza mientras estaba cerrada, y el README decía que el
agarre por fricción "no es fiable". Se probó sin la sujeción.

**Guion original, sin sujeción, 6 posiciones:** las 6 levantan ≈ 6 cm y colocan el cubo a
0,3–0,6 cm del objetivo.

**Ventana de agarre**, desplazando el punto al que apunta el experto respecto al centro del
cubo (mm, x e y del mundo). ✔ = el cubo se levanta, ✘ = no (se queda en la mesa, error final
≈ 10 cm). Dos cubos, episodio 3 (yaw +18°) y episodio 4 (yaw −28°), mordaza abierta a 1,0 rad:

| desplazamiento (x, y) | yaw +18° | yaw −28° |
|---|---|---|
| (−18, −12) | ✘ | ✘ |
| (−18, 0) | ✔ | ✔ |
| (−18, +8) | ✔ | ✔ |
| (−6, −12) | ✔ | ✘ |
| (−6, 0) | ✔ | ✔ |
| (−6, +8) | ✔ | ✔ |
| (+6, −12) | ✔ | ✘ |
| (+6, 0) | ✘ | ✔ |
| (+6, +8) | ✘ | ✘ |
| (+14, −12) · (+14, 0) · (+14, +8) | ✘ ✘ ✘ | ✘ ✘ ✘ |

Con el cubo de yaw +18° se midieron además estos puntos: ✔ en (0, 0), (−10, 0), (0, −10),
(−20, 0) y (−15, +15); ✘ en (+10, 0), (0, +10), (+15, +15), (−15, −15), (+15, −15), (+20, 0),
(0, +20) y (0, −20).

Lectura: la ventana mide unos 25 mm en x y 15 mm en y, **no está centrada en el cubo** y el
experto original (0, 0) estaba a unos 5 mm de su borde +x. Los cuatro puntos
x ∈ {−18, −6}, y ∈ {0, +8} levantan con los dos yaw. Abrir más la mordaza (1,6 rad en lugar
de 1,0) dio el mismo resultado en los nueve primeros puntos y se descartó.

Por eso `expert.GRASP_BIAS_M = (−10, +2) mm`: deja unos ±8–10 mm de margen alrededor del
punto al que apunta el guion. **El desplazamiento no era necesario para el profesor**: con el
punto original (0, 0) el guion también dio 10/10 en las mismas posiciones nuevas
(`results/runs/scripted_original_grasp_point_heldout_10ep.md`), porque el profesor conoce la
posición exacta. Sirve para tolerar el error de una política que la estima. Con el
desplazamiento, el guion por fricción dio **10/10** (`results/runs/scripted_bias_heldout_10ep_try1.md`), con elevación de
6,0–6,2 cm, transporte sujeto de 161–162 pasos y error final de 0,5–0,9 cm.

## 7. Cuánta precisión da el detector, y por qué parecía poca

Con una ventana de ±8–10 mm, la precisión de la percepción decide si una política que se
apoya en ella puede funcionar. Medida con un ajuste afín de (u, v) del detector a la
posición del cubo, validación cruzada de uno fuera:

| situación | error medio x / y (mm) | peor caso (mm) |
|---|---|---|
| 40 episodios grabados, fotograma fresco, **cámara moviéndose** en cada reinicio | 6,2 / 5,4 | 13 |
| 48 episodios grabados, ídem | 5,5 / 4,5 | 14 |
| 30 posiciones en un mismo episodio, sin `env.reset()` entre ellas | 0,55 / 0,54 | 1,7 |
| 30 episodios con reinicio entre ellas, **cámara congelada** | 0,4 / 0,6 | 2,3 |

El temblor no era del detector: renderizar de nuevo la misma escena mueve el centroide
0,1 px, y ninguna otra medida de la mancha (bounding box, cara superior, puntos extremos,
yaw real, ajuste cuadrático) bajaba de 5 mm. Se descubrió porque entre reinicios sale
distinta *toda* la imagen (el borde de la mesa se mueve 3–6 px) mientras que entre
colocaciones del cubo sin reinicio no. La causa es el apartado 6 del resumen.

Con la cámara congelada la percepción deja de ser el factor limitante: 2 mm de peor caso
frente a una ventana de ±8–10 mm.

## 8. Qué se cambió

| cambio | por qué |
|---|---|
| `il_metrics.py` | éxito = elevación ≥ 4 cm, transporte sujeto ≥ 1 s a < 3 cm de los dedos, y colocación a < 3 cm, en reposo |
| `13_collect_demonstrations.py` | sin sujeción cinemática por defecto; solo se guardan demostraciones que pasan el criterio físico; cámara fresca; guarda el reloj del plan y la posición del cubo; `--overwrite` aparta el dataset en vez de borrarlo |
| `il_env.set_cube_randomization` | congelar de verdad la aleatorización (cámara incluida); falla en voz alta si no encuentra ningún término; test de regresión en `tests/test_il_logic.py` |
| `il_env.refresh_camera`, `il_env.perceive_cube` | cámara fresca (dos renders) y detector medio de 4 renders |
| `16_evaluate_policy.py` | posiciones fuera del entrenamiento, `policy.reset()`, juicio físico, guion como referencia etiquetado, informe nuevo por ejecución sin sobrescritura |
| `tools/21_build_clock_cube_dataset.py` | dataset solo con reloj + posición del cubo y la acción: sin imagen ni estado del brazo |
| `tools/22_validate_offline.py` | reproduce sin simulador lo que mandaría la política en episodios que no vio |
| `expert.GRASP_BIAS_M` | punto de agarre en el centro de la ventana medida |

## 9. Hipótesis anteriores que esta medición supera

* **"La acción del experto no es función de una observación (449 pares ambiguos por
  episodio), así que ninguna política de una sola observación puede aprenderla."** Con la
  fase como entrada esos pares dejan de ser ambiguos (apartado 1: ajuste de 1–5 mm) y la
  política seguía fallando 0/5. La ambigüedad era real pero no era la causa del fallo en
  lazo cerrado. Las cifras de `tools/18_measure_ambiguity.py` siguen siendo correctas
  como medida del dataset.
* **"El VAE empuja hacia la demostración media."** No se midió ningún efecto del VAE: el
  ajuste sobre el entrenamiento es bueno.
* **"El cubo no se ve a partir de `close`, así que el detector no ayuda."** Se mantiene
  como medida (el detector hallaba una mancha en 25 600/25 600 fotogramas), pero no es el
  motivo del 0/5 del detector: ese checkpoint carecía de reloj y de fase, y por eso no sabía
  cuándo bajar.
* **"π0 aceptaría historial."** Cierto sobre el papel (`n_obs_steps=2`), pero no arranca
  en esta máquina (falta el fork de `siglip`) y no hizo falta probarlo.
