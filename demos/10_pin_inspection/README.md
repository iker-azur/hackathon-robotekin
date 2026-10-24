# Inspección de piezas: con perno / sin perno

Esta sección prepara los datos para el futuro modelo de dos cabezas. Por ahora
solo descubre y descarga imágenes y etiquetas; no construye ni entrena el modelo.
La arquitectura y los objetivos de cada cabeza se definirán en el paso de entrenamiento.

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

Los metadatos descargados desde el ordenador del usuario confirman 12 índices
de snapshots con 132 pares `pin` y 108 pares `nopin` (264 y 216 fotos de ambas
cámaras, antes de deduplicar). La sesión de desarrollo sigue sin conexión directa
a Hugging Face; la descarga de las fotos reales y su revisión visual están pendientes.

## Verificar sin red

```bash
python -m unittest discover -s demos/10_pin_inspection/tests
```

Las pruebas cubren la correspondencia de IDs y nombres, escenas con ambas clases,
imágenes sin etiqueta, conflictos, descarga en una revisión fija y fallos de red.

Referencia: [descarga de archivos con huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/download).
