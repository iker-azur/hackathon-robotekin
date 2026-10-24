# Inspección de piezas: con perno / sin perno

Esta sección descarga imágenes etiquetadas y entrena un clasificador de dos vistas:
`front` y `wrist` de la misma pieza. Ambas ramas comparten ResNet18 y producen
una única decisión: **Pieza con perno** o **Pieza sin perno**. Está integrada como
**Inspección de pernos** en el Training Studio, con procesos y resultados propios.

## Descargar

Desde la raíz del repositorio:

```bash
python -m pip install -r demos/10_pin_inspection/requirements.txt
python demos/10_pin_inspection/download_datasets.py
```

El comando abre directamente la interfaz. También puedes abrirla con:

```bash
python demos/10_pin_inspection/download_gui.py
```

Pulsa **Elegir carpeta…**, selecciona dónde guardar los archivos y pulsa
**Descargar y separar en ConPerno / SinPerno**. La ventana muestra el registro mientras
descarga y avisa si el análisis queda incompleto. El usuario inicial es `MikelRubi`.
Toda la configuración se hace en la ventana: usuario, destino y etiquetas adicionales
de con/sin perno. Los campos de etiquetas son opcionales; para indicar varias,
sepáralas con `;`. Se añaden a los nombres reconocidos de forma predeterminada.
Puedes escribir una ruta nueva: el descargador creará la carpeta. Cerrar la ventana
durante una descarga detiene el proceso; puedes volver a iniciarlo en el mismo
destino para reutilizar los archivos ya descargados.

La interfaz utiliza Tkinter, incluido normalmente con Python. En Ubuntu/Debian,
si falta, instálalo con `sudo apt install python3-tk`.

El propietario predeterminado es `MikelRubi`. Usa la sesión de `hf auth login`
o la variable `HF_TOKEN` para incluir los datasets privados a los que tengas acceso.
No guardes el token en el código.

El script lista los datasets accesibles del usuario y descarga primero sus JSON,
YAML y listas de clases. Las fotos del SO-101 están en `pin_snapshots/index.json`:
cada registro tiene `label` (`pin` o `nopin`), `front`, `wrist` y, opcionalmente,
`location`. El descargador conserva cámara, pareja y ubicación en el inventario,
y descarga solo `pin_snapshots/` cuando ese es el origen de las etiquetas.
Los datasets LeRobot sin snapshots etiquetados se omiten: no se asignan etiquetas
a sus vídeos ni a sus episodios por el nombre de la tarea.

También busca etiquetas explícitas como `Pieza_con_perno` y
`Pieza_sin_perno` (también `con_perno`, `sin_perno`, `with_pin`, `without_pin`,
`piece_with_pin`, `piece_without_pin`, `pin`, `nopin`, `no_pin`). Acepta diferencias
de mayúsculas, espacios, guiones y acentos. No deduce clases por la apariencia,
por ausencia de anotación ni por un ID numérico fijo.

Cuando encuentra clases objetivo en COCO, YOLO o carpetas de imágenes, descarga
el repositorio completo en la misma revisión y conserva su estructura y archivos
originales. El mecanismo de descarga de Hugging Face reutiliza archivos ya
descargados. Cada revisión tiene su propia carpeta.

Para añadir nombres exactos encontrados en tus anotaciones, utiliza los campos
**Con perno** y **Sin perno** de la interfaz. No se utilizan argumentos de terminal.

## Resultado

En `data/`, excluido de Git:

- `ConPerno/` y `SinPerno/`: copias de las imágenes con etiqueta original inequívoca,
  verificadas como archivos de imagen legibles. Conservan los bytes originales y
  usan el SHA-256 como nombre para evitar colisiones y duplicados de contenido.
- `classification_report.json`: procedencia de cada copia, recuentos, imágenes
  dañadas, escenas con etiquetas de ambas clases y conflictos pendientes. La
  separación se basa en anotaciones originales; no constituye revisión visual.
- `raw/MikelRubi/DATASET/REVISION/`: imágenes y anotaciones originales; los
  datasets sin clases reconocidas conservan solo los metadatos inspeccionados.
- `download_report.json`: datasets examinados, revisión, clases originales con
  sus IDs, recuentos y errores. `complete` requiere ambas clases, ningún error
  y ningún contenedor pendiente salvo los vídeos/tablas de datasets LeRobot
  que no tienen snapshots etiquetados.
- `inventory.jsonl`: una fila por instancia etiquetada o imagen con clase de
  carpeta, con ruta local, dataset, revisión, clase original y clase canónica
  `con_perno` / `sin_perno`. Para COCO conserva el ID de anotación; para YOLO,
  el número de línea del TXT. Las máscaras y cajas quedan en el original.

Una imagen COCO/YOLO puede contener piezas de ambas clases: sus instancias
mantienen etiquetas distintas en el inventario. Esas escenas quedan pendientes
en el informe y no se copian a ninguna clase. Tampoco se copian archivos dañados
ni fotos de contenido idéntico con etiquetas contradictorias. No crea recortes,
particiones ni aumentos de datos. El informe indica siempre que no se ha hecho
revisión visual de las fotos.

Se admiten índices `pin_snapshots/index.json`, COCO JSON, YOLO con nombres en YAML/`classes.txt`/`.names` y etiquetas
en rutas `images/…` → `labels/…`, y carpetas con nombres de clase explícitos.
Las imágenes COCO deben poder resolverse por ruta o nombre único. Los conflictos
de clases y las referencias a imágenes inexistentes se registran como errores.
Los ZIP/TAR y Parquet/Arrow sin clases reconocibles fuera de su contenedor se
enumeran en el informe y quedan pendientes de inspección; no se adivina su contenido.
El código devuelve estado de salida 1 si el análisis está incompleto.

La consulta pública verificada el 24-10-2026 descargó 350 parejas de 13 sesiones:
186 `pin` y 164 `nopin` (700 fotos), sin parejas excluidas ni duplicadas. La
partición agrupada reservó 280 parejas para entrenamiento y 70 para validación.
Estas cifras corresponden a esa consulta; se actualizarán al descargar nuevas sesiones.

## Entrenar desde el Studio

Instala las dependencias en el intérprete que vayas a seleccionar para el clasificador:

```bash
python -m pip install -r demos/10_pin_inspection/requirements-training.txt
python -m pip install -r demos/10_training_studio/requirements.txt
python demos/10_training_studio/app.py
```

No necesita LeRobot ni Isaac Sim. Conserva una pareja compatible de PyTorch y
torchvision con soporte CUDA si quieres usar la GPU; la interfaz y el intérprete
de entrenamiento pueden ser entornos distintos. La instalación no se hace
automáticamente al pulsar Entrenar.

En Windows con NVIDIA, no uses `requirements-training.txt`: PyPI instala PyTorch
sin CUDA. Cierra antes el Studio y cualquier otro proceso del `.venv`; si Windows
bloquea las DLL de PyTorch, la sustitución lo deja desinstalado a medias
(`module 'torch' has no attribute '__version__'`). Instala las compilaciones CUDA
desde la raíz del repositorio; el mismo comando repara esa instalación incompleta:

```powershell
uv pip install --link-mode=copy --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch --reinstall-package torchvision -r demos/10_pin_inspection/requirements-cuda-windows.txt
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.venv/Scripts/python.exe demos/10_training_studio/app.py
```

Esta pareja conserva los límites de LeRobot 0.6.1 (`torch<2.12`,
`torchvision<0.27`) y usa CUDA 12.6, como el entorno local de SKINEYE.
SKINEYE tiene PyTorch 2.14.0 y torchvision 0.29.0; esas versiones exceden los
límites de LeRobot y no deben copiarse al entorno compartido. En el Studio,
selecciona `.venv/Scripts/python.exe` como intérprete del clasificador y `cuda`
como dispositivo; con `auto`, un PyTorch sin CUDA entrena en CPU sin avisar.

1. Abre **Inspección de pernos → Datos**. Elige el destino local antes de descargar
   las fotos de `MikelRubi`. Para reutilizar la aplicación de descarga anterior,
   selecciona su carpeta, que debe contener `inventory.jsonl` y `raw/`, y pulsa
   **Analizar carpeta**. Se pueden actualizar las fotos en ese mismo destino.
2. Selecciona las sesiones y revisa las dos vistas de cada pareja. **Comprobar
   partición de sesiones 80/20** valida que ambas clases puedan aparecer en cada
   partición. Las carpetas `ConPerno` y `SinPerno` solas no conservan las parejas.
3. En **Entrenamiento**, elige Python y configura cámaras (`front + wrist`, solo
   `front` o solo `wrist`), épocas, lote, tasas de aprendizaje, semilla, dispositivo
   y nombre de ejecución. La cámara de cada foto sale de `pin_snapshots/index.json`,
   no se deduce de la imagen; con una sola cámara se usa esa vista de cada pareja
   completa. El checkpoint guarda sus cámaras y **Resultados** pide solo esas
   imágenes. **Aumentación de datos** (desactivada por defecto) permite elegir
   volteo horizontal y vertical, rotación (±grados), traslación (% del lado) y
   cambio de perspectiva, cada uno con su probabilidad. Se sortea de nuevo cada
   vez que se carga una imagen, por tanto en cada época, de forma independiente en
   cada cámara y reproducible con la semilla; nunca se aplica a validación ni a
   inferencia. `run_config.json` guarda los valores usados. Los valores iniciales son 30 épocas,
   16 parejas por lote, `1e-5` para ResNet18 y `1e-4` para la fusión, semilla 42 y
   dispositivo automático. El entrenamiento afina todos los pesos.
4. Pulsa **Entrenar clasificador**. Las curvas y métricas se actualizan por época.
   **Detener proceso** pide una parada entre lotes; si no responde en cinco segundos,
   se fuerza el cierre. Los checkpoints se escriben mediante archivo temporal y
   reemplazo, de modo que se conservan los últimos archivos completos.
5. En **Resultados**, selecciona una ejecución para consultar sus métricas. Elige
   una imagen `front` y una `wrist` de la misma pieza y pulsa **Clasificar pareja
   con best.pt** para obtener la clase y ambas probabilidades.

El Studio permite un solo entrenamiento pesado a la vez entre ACT y pernos.
El clasificador no modifica la selección de episodios, parámetros ni checkpoints
de ACT. Las descargas de pernos tampoco se incorporan al catálogo de datasets ACT.

### Datos, modelo y evaluación

El entrenador reconstruye las parejas desde el inventario usando dataset, revisión
y `pair_id`, con las cámaras `front` y `wrist`. Excluye parejas incompletas,
imágenes dañadas y etiquetas contradictorias. No empareja fotos sueltas de COCO,
YOLO o carpetas por parecido del nombre. Tampoco usa las etiquetas del informe
plano para adivinar qué cámara corresponde a cada archivo.

Las parejas de bytes idénticos se deduplican conservando todas sus procedencias.
Si una imagen se repite en varias sesiones, esas sesiones quedan conectadas y
se asignan al mismo lado de la partición. Se busca una separación reproducible
aproximada 80/20 que mantenga ambas clases en entrenamiento y validación. Si no
es posible, se detiene con una explicación y se requieren más sesiones: nunca
se sustituye silenciosamente por una división aleatoria de fotos.

ResNet18 utiliza pesos ImageNet, compartidos por las dos vistas. Sus vectores de
512 valores se concatenan; la fusión tiene capas `1024 → 128 → ReLU → Dropout(0.2)
→ 2`. Las imágenes se convierten a RGB, se ajustan con relleno a 224×224 sin
recortar el encuadre y se normalizan con los valores ImageNet. Entrenamiento,
validación e inferencia usan ese mismo preprocesado. La descarga inicial de pesos
debe completarse: un fallo de red no cambia el modelo a pesos aleatorios.

Se utiliza CrossEntropy y AdamW con weight decay `1e-4`. Se registran pérdida,
accuracy, precisión/recall por clase, F1 macro y matriz de confusión. Las filas
de la matriz son etiquetas reales y las columnas predicciones, en el orden
`[sin_perno, con_perno]`. **best.pt se elige por menor pérdida de validación y, si empata, por mayor
accuracy**;
esa validación participa en la selección y no es un test independiente.

### Archivos de cada ejecución

En `demos/10_pin_inspection/outputs/<nombre>/`, excluido de Git:

- `best.pt` y `last.pt`: pesos, arquitectura, preprocesado, clases, orden de cámaras,
  época y métricas. Se cargan con `weights_only=True`.
- `run_config.json`, `split.json` y `data_report.json`: parámetros, partición completa,
  procedencias, hashes, exclusiones y duplicados.
- `history.json`, `metrics.json` y `status.json`: curvas, validación del mejor modelo
  y estado `running`, `completed`, `interrupted` o `failed`.

Un nombre existente se rechaza para evitar sobrescribir entrenamientos. Una
ejecución interrumpida puede usarse para inferencia si llegó a guardar `best.pt`.
No se ofrece reanudación del optimizador ni exportación ONNX en esta sección.

Para integrar inferencia en otro código, importa `predict` de `model.py` con el
directorio del módulo en `PYTHONPATH` y llama
`predict(checkpoint, front_path, wrist_path, device="cpu")`. Devuelve `label` y
`probabilities`; las etiquetas canónicas son `sin_perno` y `con_perno`.

## Verificar sin red

La suite completa requiere las dependencias de entrenamiento y de la interfaz
del Studio indicadas arriba. Para comprobar únicamente el descargador con sus
dependencias básicas, añade `-p test_download_datasets.py` al comando.

```bash
python -m unittest discover -s demos/10_pin_inspection/tests
```

Las pruebas cubren la correspondencia de IDs y nombres, escenas con ambas clases,
imágenes sin etiqueta, conflictos, descarga en una revisión fija y fallos de red.
También comprueban emparejamiento, aislamiento de sesiones, deduplicación, un
entrenamiento breve en CPU, actualización del extractor, parada, recarga del
checkpoint, inferencia y procesos reales de Qt. Las pruebas usan pesos aleatorios
explícitos para no necesitar red; la interfaz siempre comienza con pesos ImageNet.

Referencia: [descarga de archivos con huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/download).
Arquitectura: [ResNet18 de torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Verificación real del 24-10-2026: una época en CPU con pesos ImageNet, 280 parejas
de entrenamiento y 70 de validación, terminó y permitió recargar `best.pt` para
inferir. Obtuvo accuracy de validación **54,29 %**, F1 macro **0,4017** y matriz
`[[2, 31], [1, 36]]`. Es una prueba del flujo completo, con resultados insuficientes
para dar por validada la calidad del clasificador; requiere entrenamiento completo
y evaluación posterior. No se publicaron datos ni modelos. Las fotos utilizadas
eran públicas y se descargaron con el mismo código que emplea la sección.

Suite final: **31 pruebas pasadas en 150,73 s**. Se revisaron capturas de las
tres pestañas del Studio y pasaron la compilación de Python, el análisis estático
de los módulos nuevos y la comprobación de espacios del diff.
