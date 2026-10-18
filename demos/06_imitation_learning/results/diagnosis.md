# Diagnóstico: por qué la política no completa la tarea

Fecha: 2026-10-18. Entorno `hackathon-robotekin`, Windows, Isaac Sim 5.1, RTX 5060 Ti.

Este documento recoge lo que se ha **medido** sobre el dataset y sobre la política
entrenada. Cada número sale de los artefactos que hay en el disco (el parquet, el
vídeo y las estadísticas que el propio checkpoint guardó), no de los comentarios
del código. Varias de las hipótesis que parecía razonables se han descartado por
medición, y están aquí para que no se vuelvan a perder tiempo con ellas.

## Resumen

El dataset es correcto, el objetivo está bien medido y la señal visual existe. La
política nonetheless **no hace nada**: se queda clavada emitiendo el mismo comando
durante los 800 pasos del episodio. La causa no es que la imagen no tenga
información, ni que falten demostraciones, ni que la inferencia sea de lazo
abierto. Es que la configuración de entrenamiento no consigue aprender un mapa que
un modelo **lineal** resuelve de forma casi perfecta.

## 1. Lo que el dataset sí hace bien

El profesor (`expert.py`) graba la orden de cinemática inversa, y esa orden es una
función **afín exacta** de la posición del cubo:

| fotograma | rango de `action_x` | rango de `action_y` | correlación con el cubo |
|---|---|---|---|
| 0 | 0.0007 m | 0.0006 m | −1.000 |
| 90 | **0.0700 m** | **0.0600 m** | −1.000 |
| 350 | 0.0700 m | 0.0600 m | −1.000 |
| 450 | 0.0700 m | 0.0600 m | −1.000 |

El rango del cubo es exactamente 0.0700 m en x y 0.0600 m en y. La relación es
`action_x = 0.357 − cube_x`. Es decir: **la acción graba perfectamente dónde está el
cubo**. No hay ruido ni ambigüedad en la etiqueta.

Además, el dataset **sí esMarkov** respecto a lo que se grabó. Dentro de cada
episodio hay 351 923 pares de fotogramas con `|Δestado| < 1°`; de ellos, solo el
**0.4 %** discrepa en más de 1 cm del objetivo en z, y el **0 %** cambia de signo
la velocidad en z. La hipótesis de "el profesor guionizado depende de estado
oculto y por eso ninguna política reactiva puede aprenderlo" queda **descartada**.

## 2. La señal visual existe y es fuerte

Regresión ridge (lineal) desde una imagen reducida a 24×32 en escala de grises,
sobre 275 fotogramas de la ventana "go above cube":

| λ | R² `action_x` | R² `action_y` | R² `cube_x` |
|---|---|---|---|
| 1 | **0.995** | **0.999** | **0.998** |
| 10 | 0.947 | 0.993 | 0.978 |
| 100 | 0.627 | 0.921 | 0.823 |

Un modelo **lineal** recupera la posición del cubo desde los píxeles casi
perfectamente. La información está ahí y es sencilla. Por lo tanto **no** tiene
sentido decir que "la imagen es la entrada equivocada": lo que falla es que ACT no
la aprovecha.

## 3. Lo que hace la política

Evaluando el checkpoint con el lazo cerrado (`--action-steps 4`, observando cada 4
pasos) el comando **no cambia** a lo largo de 800 pasos:

```
step  40 [go above cube] xyz=(+0.052,-0.262,+0.199)  cube=(0.302,-0.380,0.056)
step 800 [past plan]      xyz=(+0.052,-0.262,+0.198)  cube=(0.302,-0.380,0.056)
```

Y lo que es determinante: el cubo se coloca en tres x distintos y el comando
**casi no se mueve**:

| `cube_x` | comando `x` emitido |
|---|---|
| 0.302 | +0.052 |
| 0.335 | +0.048 |
| 0.369 | +0.056 |

Un recorrido de 6.7 cm del cubo produce 8 mm de variación del comando, y no con el
signo correcto. **La política no está leyendo dónde está el cubo.**

## 4. Hipótesis descartadas

Se enunciaron y se midieron; ninguna se sostiene.

* **"Las imágenes están rotas"** — falso. Decodificando el AV1, la diferencia
  media absoluta entre fotogramas de fases distintas es **11.3**, y **11.3** entre
  episodios en la misma fase. El vídeo tiene movimiento de sobra.
* **"La normalización visual está rota (`std ≈ 0.001`)"** — falso, y era un error
  de lectura. Las estadísticas reales por canal son media 0.449 y **std 0.226**.
  El `≈0.001` era la dispersión entre los tres canales, no la desviación del
  píxel. La normalización visual es correcta.
* **"Cuatro de las ocho dimensiones están constante y el desnormalizador divide
  entre cero"** — las cuatro lo están (`action.std = 0` para las dims 3-6), pero la
  red las aprende **exactamente**: la traza da `quat=(0,0,0)` y `g6=-1.000` en
  todos los pasos. El multiplicador 1e8 nunca llega a dispararse. El riesgo era
  real en el papel y falso en la práctica.
* **"El fallo es la inferencia de lazo abierto"** — real como defecto
  (`n_action_steps = 20 = chunk_size`, 32 trozos encadenados por episodio) pero
  **no la causa**: cerrar el lazo a 4 pasos da el mismo 0 %.

## 4 bis. La causa raíz, medida (2026-10-18, segunda ronda)

La ronda anterior arregló el dataset (bucle fijo, reinicio del brazo, espacio de
trabajo) y **el freeze se fue**, pero la tasa siguió en 0 %. Lo que queda es más
interesante, y es un problema del plan, no del entrenamiento.

**La tarea grabada no es función de la observación.** El brazo vuelve a pasar por
la *misma* configuración en fases distintas:

| fotogramas | separación | fases | objetivos que exigen |
|---|---|---|---|
| 299 vs 300 | 8e-6° | `lift` → `carry` | 11.2 mm distintos |
| 391 vs 420 | 0.0° | `carry` → `put down` | 9.6 mm distintos |

Contando todos los pares de fotogramas de un episodio que comparten pose dentro
de 1° pero piden objetivos separados más de 1 cm, salen **449 pares ambiguos por
episodio** (6220 en las 40 demostraciones).

Con una sola observación el modelo ve 6 números y debe devolver 8, y para esos 449
pares la mejor respuesta posible es la **media** de las dos opciones: un objetivo
que no va a ningún sitio. Eso es exactamente el rollout medido — el comando se
congelaba en un punto medio.

Añadir la velocidad del brazo **no** lo resuelve del todo (6220 → 4625 pares):
el plan repite poses en puntos distintos donde incluso la velocidad coincide.

### Por qué LeRobot no lo puede arreglar tal cual

`ACTConfig.n_obs_steps` existe pero **LeRobot 0.4.4 lo prohíbe**:

```
configuration_act.py:149
if self.n_obs_steps != 1:
    raise ValueError(f"Multiple observation steps not handled yet. ...")
```

El campo queda expuesto como `--obs-steps` en el demo 15, documentado y con
default 1, para que quede claro que es una limitación de la pila y no un descuido.

### Las dos formas de arreglarlo

1. **Que el plan no repita poses** — que `carry` termine más allá de donde
   empezó `lift`, o que el portillo del cubo cambie de sitio entre fases. Es un
   cambio en `expert.py` y no obliga a reentrenar nada más.
2. **Usar una pila que acepte historial** — `diffusion` o `pi0` de LeRobot sí
   manejan varias observaciones. Habría que comprobarlo contra la versión
   instalada antes de prometerlo.

## 5. Causa raíz provisional (ronda anterior)

La política tiene un ResNet-18, 4 capas de transformer y un cuello de botella VAE.
No consigue aprender un mapa que un modelo lineal resuelve con R² = 0.995. El
sospechoso principal es el **VAE**: con 25 demostraciones casi idénticas, el
término KL empuja hacia la demostración media, que es exactamente la única cosa
que varía. El flag `--no-vae` ya existía en `15_train_act.py` y su documentación
ya describía este riesgo, pero **el checkpoint que se evaluaba se entrenó con
`use_vae=True`**.

Pendiente de confirmar con el entrenamiento comparativo en curso.

## 6. El espacio de trabajo está empobrecido (independiente de lo anterior)

`sample_cube_spot` recorre x en línea de 0.30 a 0.37 y alterna y entre **solo dos
valores** (−0.38 y −0.44). El índice de episodio, x y el yaw van todos 1:1.

Consecuencias medidas:

* 25 "demostraciones variadas" son 25 puntos sobre una recta, no una malla.
* Una homografía plano→imagen (8 grados de libertad) **no se puede ajustar** con
  dos filas paralelas: sale con 21 px de error de partida y 427 px fuera de
  muestra. `cube_detect.Calibration.fit` lo **detecta y se niega** en vez de
  devolver una respuesta falsa.
* El brazo medido llega a 0.33 m y la caja usa 7 × 6 cm de eso.

## 7. Lo que sí funciona: el detector

`cube_detect.py` localiza el cubo por HSV y devuelve el **centroide en píxeles**,
sin calibración. Medido sobre las 100 fotogramas de demostración (`tools/17_validate_detector.py`):

| medida | valor |
|---|---|
| tasa de detección | 100 / 100 |
| forma de cubo | 100 / 100 |
| área de la mancha | 938 px de media (814–1030) |
| repetibilidad, mismo cubo | 0.9 px mediana, 6.8 px peor caso |
| linealidad en x | R² = 0.95–0.97 |
| residuo | ≤ 5.3 px  (**≤ 10 mm**) |

La escena es clara y poco saturada (saturación media 0.11) y **la mesa está en
tono 40–60°**, así que un test ingenuo de "¿es rojizo?" se dispara en el 61 % del
fotograma. Lo que lo hace funcionar es el filtro de saturación: el cubo son ~900
píxelesstrongly saturados frente a los ~13 600 de la mesa.

Con 10 mm de error y una tolerancia de colocación de 30 mm, hay margen de sobra
para alimentar la política con el centroide.

## 8. Qué se ha cambiado en el código

| cambio | por qué |
|---|---|
| `il_env.reset_arm()` | `env.reset()` no restaura las articulaciones: el episodio N empezaba donde acabó el N−1, y la pose derivaba de forma monótona (codo −39.4° → −64.5° a lo largo de 25 episodios). |
| yaw restaurado en `place_cube` | la evaluación siempre escribía el cubo alineado, así que solo probaba yaw = 0°, el caso más fácil del barrido de entrenamiento. |
| `set_cube_randomization` duplicado eliminado | estaba definido dos veces en `il_env.py`; la segunda ocultaba a la primera. |
| `--trace` / `--trace-every` en demo 16 | permite atribuir la fase en la que el brazo deja de avanzar, en vez de inferirlo. |
| Aviso de lazo abierto en demo 16 | avisa cuando `n_action_steps >= chunk_size`, que es como se entrenó el checkpoint. |
| `cube_detect.py` + `tools/17_validate_detector.py` | percepción analítica medida, sin necesidad de reentrenar nada para tenerla. |