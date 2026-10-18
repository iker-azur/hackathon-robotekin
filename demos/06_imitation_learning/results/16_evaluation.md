Fecha: 2026-10-18. Rama: prepare-demos. Entorno Conda hackathon-robotekin, Windows, Isaac Sim 5.1, RTX 5060 Ti.

Politica evaluada: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act_cube_detect\checkpoints\last\pretrained_model` (tipo `act`).
Dataset: `so101_lift_cube_cube_detect`, 40 episodios y 25 600 fotogramas. La entrada `observation.environment_state` es `[u_normalizada, v_normalizada, visible]`; el detector dio detección en 25 600/25 600 fotogramas.
Entrenamiento: 20 000 pasos, 39 min 34 s; pérdida supervisada final 0,055.
Ejecución más reciente: 1 episodio, límite de 200 pasos, `--trace --trace-every 20`. Éxito: 0/1.

Un episodio cuenta como exito si el cubo acaba a menos de 3 cm del punto donde las demostraciones lo dejaban. La altura maxima se anota al lado porque un cubo lanzado lejos ya ha fallado, aunque acabe en el sitio.

| episodio | exito | error final (cm) | altura maxima del cubo (cm) | pasos |
|---|---|---|---|---|
| 0 | no | 10.1 | 5.6 | 200 |

El registro por fase muestra que al entrar en `go down` la predicción de Z seguía
cerca de +0,199 m (altura de aproximación); no descendió hacia el cubo. X/Y sí
cambiaban. Esto apunta a una transición de acción que la política no aprendió,
no a un proceso de simulación que se haya detenido.

Evaluación completa anterior del mismo checkpoint: 0/5; errores finales 10,2,
9,8, 10,3, 10,1 y 9,5 cm; altura máxima 5,6 cm en los cinco episodios.

## Prueba anterior: ACT con fase

Checkpoint: `outputs/15_act_phase/checkpoints/last/pretrained_model`, entrenado
20 000 pasos sobre `so101_lift_cube_phase` (40 episodios, 25 600 fotogramas).
La fase se mapeó a `observation.environment_state`, el campo que ACT procesa.
Resultado: 0/5; errores finales 12,3, 10,0, 6,6, 10,0 y 10,8 cm; alturas
máximas 5,6, 6,0, 12,1, 5,6 y 6,2 cm.
