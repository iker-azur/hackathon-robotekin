# Registro de evaluaciones del demo 16

Este archivo es el registro **curado**: escrito a mano, con los números que cita el
README. Ya no lo escribe el demo 16. Cada ejecución del demo 16 deja su propio informe
(`runs/<nombre>.md` y `.json`, más `<nombre>.partial.jsonl` con un episodio por línea
escrito según termina) y **se niega a sobrescribir** uno existente, así que una prueba
corta no puede borrar una evaluación completa. [`SUMMARY.md`](SUMMARY.md) tabula todas las
ejecuciones y se regenera con `tools/23_summarize_results.py`.

Rama `prepare-demos`, entorno `hackathon-robotekin`, Windows 11, Isaac Sim 5.1, RTX 5060 Ti.
Fecha de las medidas: 2026-10-19 y 2026-10-20. Todas con `HF_HUB_OFFLINE=1`, sin
sujeción cinemática del cubo, juzgadas por `il_metrics.py` (levantar ≥ 4 cm, sujetar ≥ 1 s
a < 3 cm de los dedos, dejar a < 3 cm del objetivo y en reposo).

## Política del demo: ACT con reloj del plan + posición del cubo

Entradas: **reloj del plan (privilegiado)** y centroide del cubo en píxeles medido una vez al
empezar (detector de color). Sin imagen ni estado articular en la red. Entrenada 10 000
pasos sobre 48 demostraciones (`outputs/15_act_clock`).

| ejecución | cubos | éxitos | notas |
|---|---|---|---|
| [`act_clock_final_heldout_10ep`](runs/act_clock_final_heldout_10ep.md) | 10 posiciones nuevas entre las de entrenamiento | **10/10** | elevación 5,9–6,1 cm, transporte 158–167 pasos, error final ≤ 1,1 cm |
| [`act_clock_final_heldout_from10_20ep`](runs/act_clock_final_heldout_from10_20ep.md) | otras 20 posiciones nuevas | **20/20** | elevación 5,9–6,5 cm, error final ≤ 0,9 cm |
| [`act_clock_final_heldout_10ep_repeat`](runs/act_clock_final_heldout_10ep_repeat.md) | las mismas 10 de la primera | **10/10** | repetición para ver la variación de la física |
| [`act_clock_val_40demos_heldout_10ep`](runs/act_clock_val_40demos_heldout_10ep.md) | 10 posiciones nuevas | **10/10** | política entrenada solo con los episodios 0–39 |
| [`act_clock_zip_roundtrip_heldout_2ep`](runs/act_clock_zip_roundtrip_heldout_2ep.md) | 2 posiciones nuevas | **2/2** | el modelo comprimido y descomprimido en otra carpeta |
| [`act_clock_final_outside_10ep`](runs/act_clock_final_outside_10ep.md) | **fuera** de la región de entrenamiento (anillo de 1 cm) | **8/10** | fallos: `misplaced` (cubo a 8,2 y 5,0 mm fuera de la caja, error 3,2 cm) y `never_lifted` (a 9,0 mm fuera) |

**Lo que dicen estas cifras y lo que no.** 30 posiciones distintas, 30 aciertos (40/40 con la
repetición). Pero esas posiciones están **intercaladas** con las 48 demostraciones: la más
cercana está a 4,6–7,2 mm. Eso prueba interpolación, no extrapolación. Fuera de la región
el éxito baja (8/10), y los dos fallos son las posiciones más lejanas (20–25 mm de la
demostración más cercana, 8–9 mm fuera de la caja). Una primera versión de esa prueba
([`act_clock_final_wide_10ep`](runs/act_clock_final_wide_10ep.md), 10/10) muestreaba toda
la caja ensanchada y solo 2 de 10 posiciones cayeron fuera, por menos de 3 mm; no mide
extrapolación y se conserva solo por completitud.

## Referencia: el guion del profesor (no es una política aprendida)

Sabe dónde está el cubo. Sin sujeción cinemática.

| ejecución | cubos | éxitos |
|---|---|---|
| [`scripted_bias_heldout_10ep_try1`](runs/scripted_bias_heldout_10ep_try1.md) | 10 nuevas, punto de agarre desplazado (−10, +2) mm | **10/10** |
| [`scripted_original_grasp_point_heldout_10ep`](runs/scripted_original_grasp_point_heldout_10ep.md) | las mismas 10, punto de agarre original (0, 0) | **10/10** |
| [`scripted_outside_10ep`](runs/scripted_outside_10ep.md) | las mismas 10 de fuera de la región | **10/10** |
| [`scripted_wide_10ep`](runs/scripted_wide_10ep.md) | las de la primera versión de "wide" | **10/10** |

El desplazamiento del punto de agarre no era necesario para el profesor (acierta con ambos
puntos); da margen a una política que estima la posición del cubo.

## Políticas anteriores con cámara + estado articular, con el protocolo nuevo

5 posiciones nuevas cada una, mismo juicio físico. Son los checkpoints de la evaluación original,
sin reentrenar; se evalúan ahora con la cámara congelada.

| ejecución | entradas | éxitos | fallos |
|---|---|---|---|
| [`baseline_act_image_state_heldout_5ep`](runs/baseline_act_image_state_heldout_5ep.md) | imagen + estado | **0/5** | `never_lifted` ×5 (el cubo ni se toca: error final ≈ 10 cm) |
| [`baseline_act_cube_detect_heldout_5ep`](runs/baseline_act_cube_detect_heldout_5ep.md) | imagen + estado + centroide del cubo | **0/5** | `never_lifted` ×5 (ídem) |
| [`baseline_act_phase_heldout_5ep`](runs/baseline_act_phase_heldout_5ep.md) | imagen + estado + **fase (privilegiada)** | **0/5** | `never_lifted` ×4, `misplaced` ×1 (un episodio levantó 6,4 cm y lo dejó a 6,9 cm) |

El diagnóstico de por qué fallan está en [`diagnosis.md`](diagnosis.md).

## Sin simulador

[`offline/act_clock_val_10k_heldout_40-47.json`](offline/act_clock_val_10k_heldout_40-47.json):
política entrenada con los episodios 0–39, reproducida sobre los 8 que no vio; error de posición
del comando mientras se cierra la pinza, 3,4 mm de media y 5,0 mm de peor caso en xy, 0,3 mm en z.

## Lo que no está medido

* **La política no se ha probado fuera del simulador** ni con otra fricción.
* **La parada por bloqueo** (`--stall-seconds`, escribe lo terminado y sale con código 3) **no se ha
  disparado nunca**: se añadió tras un bloqueo de Kit en una ejecución de 20 episodios y todas las
  posteriores terminaron sin necesitarla. Está sin probar.
* **Ctrl+C** escribe el informe parcial por diseño, pero no se ha probado con Kit en Windows.

## Historial: evaluaciones con el protocolo antiguo (superado)

Transcrito del commit `a3db19f` (el original, con la ruta absoluta de la máquina, está en
`git show a3db19f:demos/06_imitation_learning/results/16_evaluation.md`). Ese protocolo evaluaba
**las mismas posiciones que las demostraciones**, juzgaba solo la posición final del cubo y
sobrescribía este archivo en cada ejecución; no lo uses como referencia.

> Fecha: 2026-10-18. Rama: prepare-demos. Entorno Conda hackathon-robotekin, Windows, Isaac Sim 5.1, RTX 5060 Ti.
>
> Politica evaluada: `outputs\15_act_cube_detect\checkpoints\last\pretrained_model` (tipo `act`).
> Dataset: `so101_lift_cube_cube_detect`, 40 episodios y 25 600 fotogramas. La entrada `observation.environment_state` es `[u_normalizada, v_normalizada, visible]`; el detector dio detección en 25 600/25 600 fotogramas.
> Entrenamiento: 20 000 pasos, 39 min 34 s; pérdida supervisada final 0,055.
> Ejecución más reciente: 1 episodio, límite de 200 pasos, `--trace --trace-every 20`. Éxito: 0/1.
>
> | episodio | exito | error final (cm) | altura maxima del cubo (cm) | pasos |
> |---|---|---|---|---|
> | 0 | no | 10.1 | 5.6 | 200 |
>
> El registro por fase muestra que al entrar en `go down` la predicción de Z seguía cerca de +0,199 m
> (altura de aproximación); no descendió hacia el cubo. X/Y sí cambiaban.
>
> Evaluación completa anterior del mismo checkpoint: 0/5; errores finales 10,2, 9,8, 10,3, 10,1 y 9,5 cm;
> altura máxima 5,6 cm en los cinco episodios.
>
> Prueba anterior: ACT con fase (`outputs/15_act_phase`, 20 000 pasos sobre `so101_lift_cube_phase`):
> 0/5; errores finales 12,3, 10,0, 6,6, 10,0 y 10,8 cm; alturas máximas 5,6, 6,0, 12,1, 5,6 y 6,2 cm.
