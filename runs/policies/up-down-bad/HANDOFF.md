# Política ACT «Up-down BAD»: traspaso al agente del robot

Esta carpeta es **el punto de entrega fijo** de la política que lleva la pieza desde la cámara de inspección hasta la **caja «bad»**. «Bad» es solo el nombre de esa caja física en los datos (la otra es «good»); no significa que la pieza sea defectuosa. Lo que hay en `latest/` es siempre el mejor checkpoint disponible; se sobrescribe cuando el entrenamiento produce uno mejor (mira `latest/manifest.json` → `training_step` y `status`).

```
runs/policies/up-down-bad/
├── HANDOFF.md                 ← este documento
├── run_policy.py              ← bucle de control listo para el SO-101 (y modo --dry-run sin hardware)
└── latest/
    ├── pretrained_model/      ← la política (ACTPolicy.from_pretrained) + normalizadores
    ├── dataset_features.json  ← nombres/orden/forma de entradas y salidas (sin necesitar el dataset)
    ├── manifest.json          ← qué es, de dónde sale, estado y objetivos
    ├── eval_heldout.json      ← métricas en episodios que no vio al entrenar
    ├── eval_bad.py            ← script de esa evaluación
    └── train_wrapper.py       ← lanzador de lerobot-train usado (Windows)
```

## 1. Dónde encaja en el sistema completo

```
[Política 1: pick]  coge la pieza y la eleva delante de la cámara de inspección
        │
[Selector de destino]  decide a qué caja va la pieza
        │   · hoy (ejemplo): el clasificador de perno (vistas front + wrist,
        │     demos/10_pin_inspection) elige la caja según «con perno» / «sin perno»
        │   · futuro: el operario elige la caja según sus necesidades
        │
        ├── caja «good» → [Política Up-down GOOD]   (otra política, no es esta)
        └── caja «bad»  → [Política Up-down BAD]    ← ESTA
```

Cada política de Up-down es **un recorrido a una caja concreta**, no una decisión de calidad. Esta no sabe por qué la pieza va a la caja «bad», ni lee el clasificador ni al operario: solo ejecuta el trayecto desde la cámara hasta esa caja. El orquestador del robot mantiene el mapa *destino elegido → política* y la invoca. Así, cambiar el criterio de elección (clasificador, operario, pedido…) no obliga a reentrenar nada; solo añadir una caja nueva requiere grabar y entrenar su propia política de recorrido.

## 2. Contrato: qué espera al empezar y cómo queda al terminar

Medido sobre las 88 demostraciones válidas (unidades del follower de LeRobot: grados; pinza 0–100):

| | pan | lift | elbow | wrist_flex | wrist_roll | pinza |
|---|---|---|---|---|---|---|
| **Inicio** (media) | −33 | 1 | −38 | 1 | 114 | 5 |
| Inicio (desv. típ.) | 9 | 26 | 43 | 33 | **61** | 5 |
| Inicio (mín … máx) | −49…−12 | −64…41 | −96…50 | −73…78 | −58…167 | 1…26 |
| **Final** (media) | 6 | −100 | 29 | 100 | −5 | 1 |
| Final (desv. típ.) | 8 | 10 | 10 | 4 | 9 | 1 |

- **Precondición:** la pieza ya está **sujeta en la pinza (cerrada, ≈5)** delante de la cámara de inspección, en la pose en que la deja la política de pick. Las poses de inicio son muy variadas (sobre todo `wrist_roll`), así que no hace falta llevar el brazo a una pose exacta; pero si la pose de traspaso cae fuera de los rangos de arriba, la política está fuera de lo que vio.
- **Lo que hace:** lleva la pieza a la caja «bad», **abre la pinza (≈35, hacia la mitad del recorrido)**, suelta, y vuelve a la **pose de reposo** con la pinza cerrada.
- **Duración:** mediana 6,7 s (p10 5,3 s, p90 8,6 s, máx. 20 s).
- **Condición de fin** (ACT no emite «he terminado»): `run_policy.py` para cuando la pinza **ya se abrió por encima de 20** y el estado está dentro de la tolerancia de la pose final (`REST_POSE ± REST_TOLERANCE` en el script), o tras `--max-seconds` (15 s por defecto).

## 3. Entradas y salidas exactas

| Clave | Contenido |
|---|---|
| `observation.images.front` | cámara frontal / de inspección, RGB 480×640, 30 fps |
| `observation.images.wrist` | cámara de muñeca, RGB 480×640, 30 fps |
| `observation.state` | 6 posiciones medidas del follower |
| `action` (salida) | 6 consignas **absolutas** de posición, mismo orden |

Orden articular: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (claves `*.pos` del `SO101Follower`). Robot `so101_follower` con `use_degrees=True` (valor por defecto en LeRobot 0.6.1). La normalización va dentro de `pretrained_model/` (pre/post-procesadores): se le pasan valores crudos.

**Las cámaras deben llamarse exactamente `front` y `wrist`** en la configuración del robot, y estar montadas, encuadradas e iluminadas como en la grabación. Si el grabador del PC del robot usa otros nombres (p. ej. `overhead`), hay que mapearlos a estos dos. Una cámara cambiada de sitio es la forma más rápida de que la política falle.

## 4. Cómo ejecutarla

Requiere **LeRobot 0.6.1** (con el que se grabaron los datos y se entrenó). Desde esta carpeta:

```bash
# 1) Sin hardware: comprueba el entorno y el modelo con un episodio grabado
python run_policy.py --dry-run-dataset <ruta>/up-down-bad_all --episode 4

# 2) En el robot (MUEVE EL BRAZO): pieza ya sujeta delante de la cámara
python run_policy.py --port <puerto_follower> --front-camera <idx_o_ruta> --wrist-camera <idx_o_ruta> \
    --robot-id <id_de_la_calibración> [--calibration-dir <dir>]
```

Para integrarlo en el orquestador, usa las funciones de `run_policy.py`: `load_policy()` una vez al arrancar, y por cada pieza enviada a la caja «bad» `policy.reset()` → bucle `step(observation)` → `robot.send_action(...)` a 30 Hz hasta `finished(...)`. **Llama a `policy.reset()` antes de cada pieza**: ACT guarda una cola de acciones pendientes y, si no se vacía, la siguiente ejecución arranca con acciones de la anterior.

Parámetros de seguridad y control:

- `--max-relative-target 5`: limita a 5° (o 5 puntos de pinza) el salto por consigna, igual que en la grabación. Para las primeras pruebas, déjalo así y ten la mano en la parada.
- `--n-action-steps 25`: re-observa cada 25 acciones (0,8 s) en lugar de ejecutar el bloque entero de 100. En la evaluación offline el error crece con el horizonte (≈5 en el paso 0 → ≈17 en el paso 99), así que replanificar más a menudo debería ayudar; bajar a 10 da un lazo aún más cerrado a costa de más cómputo.

## 5. Estado actual y cómo leerlo

`latest/manifest.json` indica el paso de entrenamiento y si el checkpoint cumple los objetivos. Métricas en 9 episodios no vistos (4, 14, 24, 35, 45, 55, 65, 75, 85), en `latest/eval_heldout.json`:

| Objetivo de aceptación | Umbral |
|---|---|
| Mejora del error frente a «copiar el estado» | ≥ 50 % |
| Error del primer paso de acción | < 3 |
| F1 de apertura de pinza (pinza > 30) | ≥ 0,7 |

Son medidas **offline** (predicción frente a la acción del experto). **No sustituyen probarla en el robot**: ninguna de estas políticas se ha ejecutado todavía en el brazo real. La señal definitiva es: de N piezas enviadas a la caja «bad», cuántas acaban dentro y el brazo vuelve a reposo.

## 6. Procedencia (para reentrenar o depurar)

- **Datos:** los 20 datasets `MikelRubi/up-down-bad_*` del Hub (tarea «Carry the bracket from the camera to the bad box and release it»), unidos en `demos/09_voice_bin_picking/data/up-down-bad_all` (91 episodios, 30 fps). Excluidos por ser demostraciones defectuosas: 28 (invertido, abre la pinza 4 veces), 32 (cortado, no vuelve a reposo), 41 (no vuelve a reposo). 79 para entrenar, 9 para evaluar.
- **Entrenamiento:** `lerobot-train` 0.6.1, ACT por defecto (ResNet18 ImageNet, chunk 100, VAE, lr 1e-5), lote 8, aumentos de imagen activados, AMP. Salida completa con estado del optimizador en `runs/Up-down/Bad/act_bad_v1/checkpoints/<paso>/`.
- **Continuar:** `python latest/train_wrapper.py --resume=true --config_path=runs/Up-down/Bad/act_bad_v1/checkpoints/<paso>/pretrained_model/train_config.json` (desde la raíz del repo). El wrapper solo evita el symlink `last` que Windows sin privilegios no permite crear.
