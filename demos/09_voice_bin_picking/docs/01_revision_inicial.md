# 1 · Revisión inicial del proyecto

Qué había, qué funcionaba, qué faltaba y qué se decidió reutilizar, antes de escribir nada. Todo lo de esta tabla se comprobó leyendo el código y los
informes del repositorio (rama `prepare-demos`, 2026-10-21); donde dice «medido» la cifra es de los informes de cada demo, no de esta revisión.

## Lo que hay

| componente | dónde | qué es | estado real |
|---|---|---|---|
| Simulación básica del SO-101 | `01_simulation_basics` | LeIsaac + Isaac Lab 2.3, cubo, pick & place | funciona; punto de partida del demo 06 |
| Voz | `02_voice`, `05_planner_and_executor/voice.py` | `faster-whisper` (modelo *small* en la caché local) + `sounddevice`; el `Listener` graba hasta el silencio y transcribe | funciona; **se reutiliza** |
| Intención | `03_command_model`, `05_.../command_model_v2`, `intent_parser.py` | GLiNER2.5 ajustado (136 frases, 84,6 % de órdenes exactas en su propio informe) + reglas | funciona para «pon el cubo rojo en la caja roja»; **sus destinos son bandejas de color, no izquierda/derecha**; pesa cientos de MB y, cargado tras arrancar Isaac, rompe el proceso. **No se reutiliza** (decisión en `02_arquitectura.md`) |
| Percepción | `04_perception`, `05_.../perceive.py`, `08_recovery_kit/recoverykit/perception.py` | color saturado (HSV) + contornos + profundidad. El demo 08 mide 98,5 % de recall y ~1 mm de error en 20 imágenes de 7 piezas | funciona con piezas **sueltas**; **no separa piezas pegadas o apiladas** y no entrega máscaras por instancia |
| **YOLO** | — | — | **No existe.** No hay pesos, ni `ultralytics` en el entorno, ni datos de entrenamiento. El informe del demo 11 (`demo11_plan_mejoras.md`) recomendaba *no* usarlo hasta demostrar que HSV no llegaba; este demo es justo ese caso (piezas amontonadas) |
| Aprendizaje por imitación | `06_imitation_learning` | datasets LeRobot v3, ACT, evaluación física | ACT con *sólo* `[reloj del plan, posición del cubo]` acierta 30/30; **el reloj es privilegiado y es de lazo abierto**; ACT con imagen + estado articular: **0/5** por copiar la pose de su propio brazo (*confusión causal*, medido en `results/diagnosis.md`) |
| Taller físico con agarre por fricción | `08_recovery_kit` (**sin confirmar en git**) | robot, mesa, cajas y cámara RGB-D en Isaac con física y servos con límites de par, cinemática y control propios, agarre 30/30 en cubos y cilindros de 25 mm, recuperación de fallos, panel web | funciona (medido); **es la base de este demo** |
| Cuadernos de Colab | — | — | **No existe ninguno.** Se crean (`colab/`) |
| Interfaz | `05_.../monitor`, `08_recovery_kit/server` | paneles web con FastAPI | el del demo 11 es muy denso; el del 08 es de kits (3D propio). Se escribe uno nuevo, más simple |

## Qué faltaba para el objetivo

1. Un **contenedor con piezas amontonadas** con orientaciones aleatorias, contacto, apilado y oclusión, asentado por física (el demo 08 coloca piezas planas y separadas).
2. **Segmentación por instancias** que aguante piezas pegadas/apiladas, y color/forma/posición medidos sobre cada máscara.
3. **Dos cajas de destino** (izquierda/derecha) y órdenes que las nombren.
4. Una **política de agarre aprendida condicionada a una pieza concreta** y el flujo para entrenarla con ~500 demostraciones.
5. Una **máquina de estados** que verifique con la cámara y se recupere, y que diga **qué política hizo cada agarre**.
6. Interfaz simple en español; pruebas de todo lo anterior; cuadernos.

## Discrepancias con el enunciado (a la vista)

* «Integrar el modelo YOLO existente» → **no hay modelo**. Se entrega la tubería completa (exportación de etiquetas, cuaderno de ajuste, envoltorio de inferencia,
  evaluación contra HSV) y el demo funciona con HSV hasta que lo entrenes.
* «Reutilizar los cuadernos de Colab» → **no hay cuadernos**. Se crean dos.
* «Isaac Lab / LeIsaac» → se usa el taller del demo 08 (Isaac Lab con el USD del SO-101 de LeIsaac) porque trae física, cinemática y agarre **ya medidos**; el entorno
  `LeIsaac-SO101-*` del demo 06 no tiene nada de esto y habría que reconstruirlo.
* «Piezas industriales»: cubos y cilindros de 25 mm en tres colores (el amarillo se confunde con el brazo, que es amarillo). Son lo único con agarre validado.

## Hallazgos de la revisión que cambiaron el diseño

Medidos con la cinemática y los colisionadores reales del SO-101 (todo reproducible sin Isaac, ver `04_validacion.md`):

* **El alcance manda.** Con la pinza vertical el brazo sólo llega a todas las alturas del ciclo en una corona de 0,16–0,23 m (demo 08, `reach_map.json`). El contenedor
  (80 × 200 mm interior) y las dos cajas (115 mm) están colocados con esa corona: contenedor 75/75 puntos alcanzables en el suelo y 65/75 en la 2.ª capa; cajas 33/36 y 36/36.
* **La subida de 6,5 cm del demo 08 no sirve en pilas:** con la pieza de la 2.ª capa, **0 de 42** puntos eran alcanzables. Ahora la subida depende de la altura (`picking.lift_height`).
* **El brazo plegado del demo 08 tapaba una esquina de la caja** en la imagen cenital; se buscó otra postura de observación sin solape (`config.OBSERVE_JOINTS`). La primera elegida sólo con cinemática resultó físicamente imposible en Isaac (eslabones tocándose, brazo atascado); la actual está comprobada en el simulador (ver `04_validacion.md`).
* **Los dedos importan:** miden 8–10 mm de grosor y la mordaza móvil abre 12–16 mm; con eso, una pieza a <10 mm de una pared o pegada a otra queda **sin hueco** en uno o los dos ejes.
  Piezas en una esquina, o rodeadas, no se pueden coger sin antes mover las vecinas (límite físico, ver `03_limitaciones.md`).
* **El cuerpo de la pinza choca con las paredes** (no sólo los dedos): se comprueba con colisión de vértices (`collision.py`) y se elige un giro sin choque.
* **Capacidad real de una caja de 115 mm: 5 piezas** a la vez (no 4 por rejilla); con 105 mm eran 3.
