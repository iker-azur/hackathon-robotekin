"""Genera los cuadernos de Colab de ``colab/`` (no hay cuadernos previos en el repositorio: se crean aquí).

    python tools/build_notebooks.py

* ``colab/01_entrenar_yolo_seg.ipynb``  ajusta YOLO-seg (y compara con YOLOE por texto) con el conjunto de ``export_yolo_dataset.py``;
* ``colab/02_entrenar_act.ipynb``       entrena ACT con el dataset de ``collect_demos.py`` y lo comprueba sin simulador.

El cuaderno de ACT incrusta, tal cual, ``tools/train_act.py`` y ``tools/offline_eval_policy.py``: lo que las pruebas ejercitan en local es
lo que corre en Colab. Si cambias esos scripts, vuelve a ejecutar esto. Los cuadernos NO se han ejecutado nunca (no se lanzan entrenamientos
desde la revisión); las estimaciones de tiempo son estimaciones.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def writefile(name: str, path: Path) -> dict:
    return code(f"%%writefile {name}\n" + path.read_text(encoding="utf-8"))


# ===================================================================== YOLO
YOLO = notebook([
    md("""
# 01 · Ajustar YOLO-seg para segmentar las piezas del contenedor

**Para qué.** El demo separa las piezas con un segmentador. Por defecto usa HSV + profundidad (sin entrenar). Este cuaderno ajusta una red
**YOLO de segmentación** con imágenes de la cámara cenital simulada, para compararla con HSV con exactamente las mismas imágenes y etiquetas.
**En el repositorio no había ningún modelo YOLO entrenado**: esto lo crea.

**Entrada.** La carpeta que escribe `tools/export_yolo_dataset.py` en Isaac Sim (`data/yolo_pieces/`), comprimida en un zip y subida a Drive.
Una sola clase, *pieza*: **el color y la forma no los decide la red**, se miden en la máscara (`binpick/attributes.py`), así que la misma red sirve
con otros colores.

**Salida.** `piezas_seg.pt` (mejor modelo), que se usa con `python 18_voice_bin_picking.py --segmenter yolo:piezas_seg.pt` y se evalúa con
`python tools/eval_segmentation.py --data data/yolo_pieces --segmenter yolo:piezas_seg.pt`.

**Tiempo orientativo (sin medir).** ~400 escenas × 3 vistas ≈ 1 200 imágenes; `yolo11n-seg` a 640 px, 100 épocas: del orden de 20–40 min en una T4.
Aviso honesto: las etiquetas son analíticas y la red verá sólo este render; sobre una cámara real habría que volver a ajustarla.
"""),
    code("""
!nvidia-smi -L
import torch; print("torch", torch.__version__, "· CUDA:", torch.cuda.is_available())
"""),
    code("""
!pip -q install -U ultralytics
import ultralytics; print("ultralytics", ultralytics.__version__)   # anota la versión junto a los resultados
"""),
    md("## Datos\nSube `yolo_pieces.zip` a tu Drive (carpeta `piezas/`) o cámbialo por una subida directa."),
    code("""
from google.colab import drive
drive.mount("/content/drive")
ZIP = "/content/drive/MyDrive/piezas/yolo_pieces.zip"     # <- ajusta
!rm -rf /content/yolo && mkdir -p /content/yolo && unzip -q "{ZIP}" -d /content/yolo
import pathlib, glob
root = next(p.parent for p in pathlib.Path("/content/yolo").rglob("data.yaml"))
print("dataset:", root)
for s in ("train", "val"):
    print(s, len(glob.glob(f"{root}/images/{s}/*.jpg")), "imágenes")
print(open(root / "data.yaml").read())
"""),
    code("""
# Comprobación visual: las etiquetas deben cuadrar con las piezas
import cv2, random, matplotlib.pyplot as plt
imgs = sorted(glob.glob(f"{root}/images/train/*.jpg")); random.seed(0)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, f in zip(axs, random.sample(imgs, 3)):
    im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]
    lab = f.replace("images", "labels").replace(".jpg", ".txt")
    for line in open(lab):
        v = list(map(float, line.split()[1:])); pts = [(int(v[i] * w), int(v[i + 1] * h)) for i in range(0, len(v), 2)]
        cv2.polylines(im, [__import__("numpy").array(pts, "int32")], True, (255, 255, 0), 1)
    ax.imshow(im); ax.axis("off")
plt.show()
"""),
    md("## Entrenar\nValores de partida razonables, no medidos: el recorte de color queda **desactivado** (`hsv_h=0`) porque el color es un atributo que se mide después."),
    code("""
from ultralytics import YOLO
MODEL, EPOCHS, IMGSZ, BATCH = "yolo11n-seg.pt", 100, 800, 16
model = YOLO(MODEL)
model.train(data=f"{root}/data.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=30, project="/content/runs", name="piezas",
            hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5, degrees=0, mosaic=1.0, seed=0, workers=4, plots=True)
"""),
    code("""
best = YOLO("/content/runs/piezas/weights/best.pt")
m = best.val(data=f"{root}/data.yaml", imgsz=IMGSZ, split="val")
print("máscara  mAP50 = %.3f · mAP50-95 = %.3f" % (m.seg.map50, m.seg.map))
print("caja     mAP50 = %.3f · mAP50-95 = %.3f" % (m.box.map50, m.box.map))
"""),
    md("""
## (Opcional) YOLOE sin entrenar, por texto

Útil como punto de comparación y para piezas reales que aún no tienen datos. Necesita descargar pesos (y el codificador de texto la primera vez).
Las métricas **por instancia, color, posición y oclusión** se calculan en local con `tools/eval_segmentation.py` sobre el mismo conjunto; aquí sólo se ve a ojo.
"""),
    code("""
try:
    from ultralytics import YOLOE
    z = YOLOE("yoloe-11s-seg.pt"); names = ["industrial part", "cube", "cylinder"]
    z.set_classes(names, z.get_text_pe(names))
    r = z.predict(sorted(glob.glob(f"{root}/images/val/*.jpg"))[0], conf=0.25, imgsz=IMGSZ, retina_masks=True, verbose=False)[0]
    print("YOLOE sin entrenar detecta", 0 if r.masks is None else len(r.masks.data), "instancias en la primera imagen de val")
except Exception as e:                       # versión de ultralytics sin YOLOE o sin red
    print("YOLOE no disponible aquí:", e)
"""),
    code("""
import shutil
shutil.copy("/content/runs/piezas/weights/best.pt", "/content/piezas_seg.pt")
shutil.copy("/content/piezas_seg.pt", "/content/drive/MyDrive/piezas/piezas_seg.pt")
from google.colab import files; files.download("/content/piezas_seg.pt")
"""),
    md("""
## Usarlo en el demo

En la máquina con Isaac Sim (el entorno conda del proyecto necesita `ultralytics` **sin tocar las versiones ya fijadas** de numpy/torch):

```powershell
python -m pip install ultralytics --no-deps
python -m pip install ultralytics-thop polars   # si el import se queja de módulos que faltan; añade lo que pida sin actualizar numpy
python tools/eval_segmentation.py --data data/yolo_pieces --segmenter yolo:piezas_seg.pt
python 18_voice_bin_picking.py --segmenter yolo:piezas_seg.pt --open
```

No se ha probado ninguna instalación de ultralytics en ese entorno: si `import ultralytics` rompe numpy, vuelve a `--segmenter hsv`.
"""),
])

# ======================================================================= ACT
CHECK = '''
# Comprobación rápida del dataset (sólo lectura: claves, formas, NaN, unidades, que las acciones se mueven)
import json, glob, numpy as np, pandas as pd, pathlib
info = json.load(open(DATA / "meta" / "info.json"))
print("episodios:", info["total_episodes"], "· fotogramas:", info["total_frames"], "· fps:", info["fps"])
for k in ("observation.images.overhead", "observation.images.target", "observation.state", "action"):
    assert k in info["features"], f"falta {k}"
bi = json.load(open(DATA / "meta" / "binpick_info.json")); print("recorte:", bi["roi"], "→", bi["out_hw"], "· segmentador:", bi["segmenter"], "· ruido:", bi["noise"])
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "data/*/*.parquet")))], ignore_index=True)
S = np.stack(df["observation.state"]); A = np.stack(df["action"])
assert np.isfinite(S).all() and np.isfinite(A).all(), "NaN en estado o acción"
assert (np.abs(A[:, :5]) <= 101).all() and (A[:, 5] >= -1).all() and (A[:, 5] <= 101).all(), "acciones fuera de unidades LeRobot"
print("desviación típica de la acción por articulación:", A.std(0).round(1), "· |acción−estado| medio:", np.abs(A - S).mean().round(2))
assert np.abs(A - S).mean() > 0.02, "la acción es una copia del estado"
ep = df["episode_index"].to_numpy(); L = np.bincount(ep)
print("longitud de episodio (fotogramas): min", L.min(), "mediana", int(np.median(L)), "max", L.max())
log = DATA / "meta" / "binpick_episodes.jsonl"
if log.exists():
    rows = [json.loads(l) for l in open(log)]; ok = sum(r.get("success", False) for r in rows)
    print(f"experto: {ok}/{len(rows)} intentos válidos ({100*ok/len(rows):.0f} %)")
'''

ACT = notebook([
    md("""
# 02 · Entrenar la política ACT de agarre (LeRobot) y comprobarla sin simulador

**Qué aprende.** Dado el RGB de la cámara cenital, la **segmentación que marca qué pieza coger** y las articulaciones, ACT predice bloques de
50 consignas de articulación (aproximar → bajar → cerrar → subir). **No** aprende a llevar la pieza a la caja: eso es programado
(`docs/02_arquitectura.md`: arquitectura híbrida y por qué).

**Entrada.** `data/bin_pick_v1/` grabado con `tools/collect_demos.py` (≈ 500 episodios), comprimido en zip y subido a Drive.
**Salida.** `pretrained_model` + `binpick_info.json`, que se usa con `python 18_voice_bin_picking.py --policy act:<carpeta>`.

**Tiempo orientativo (sin medir).** 500 episodios ≈ 85 000 fotogramas; 50 000 pasos con lote 16 en una T4 son varias horas. Empieza con una
pasada corta (`STEPS = 2000`) para ver que la pérdida baja antes de lanzar la larga.

**Variantes que conviene comparar** (el mismo dataset sirve para todas): con `OBS_MODE`, `rgb_mask` (recomendada),
`rgb_only` (control: no sabe qué pieza; debería fallar con varias piezas) y `mask_only` (¿basta la segmentación sola?). y además `NO_VAE` y `NO_STATE` (sin las articulaciones como entrada: en el demo 06 la política con cámara + estado copiaba la pose de su propio brazo). No se ha medido cuál gana:
ése es el experimento.

**Importante.** Un buen error offline **no** garantiza que agarre. Lo que decide es `tools/eval_sorting.py` en Isaac.
"""),
    code("""
!nvidia-smi -L
import torch; print("torch", torch.__version__, "· CUDA:", torch.cuda.is_available())
"""),
    code("""
!pip -q install "lerobot==0.4.4"      # la versión con la que se grabó el dataset (formato v3)
import lerobot, torch; print("lerobot", getattr(lerobot, "__version__", "?"), "· torch", torch.__version__)
"""),
    md("## Datos y parámetros"),
    code("""
from google.colab import drive
drive.mount("/content/drive")
ZIP        = "/content/drive/MyDrive/piezas/bin_pick_v1.zip"    # <- ajusta
OBS_MODE   = "rgb_mask"        # rgb_mask | rgb_only | mask_only
STEPS      = 50000             # empieza con 2000 para una prueba
BATCH      = 16                # baja a 8 si se queda sin memoria
CHUNK, ACTION_STEPS = 50, 25
NO_VAE     = False             # segundo experimento: True
NO_STATE   = False             # tercer experimento: True = sin articulaciones como entrada (anti copia de la pose del propio brazo)
TRAIN_EPISODES = "0-449"       # 450-499 quedan sin ver para la comprobación offline
OUT        = f"/content/outputs/act_{OBS_MODE}"
import pathlib
!rm -rf /content/data && mkdir -p /content/data && unzip -q "{ZIP}" -d /content/data
DATA = next(p.parent.parent for p in pathlib.Path("/content/data").rglob("info.json") if p.parent.name == "meta")
print("dataset:", DATA)
"""),
    code(CHECK),
    writefile("train_act.py", ROOT / "tools" / "train_act.py"),
    code("""
flags = ("--no-vae " if NO_VAE else "") + ("--no-state" if NO_STATE else "")
!python train_act.py --dataset-root "{DATA}" --obs-mode {OBS_MODE} --steps {STEPS} --batch {BATCH} --chunk {CHUNK} --action-steps {ACTION_STEPS} --episodes {TRAIN_EPISODES} --output "{OUT}" {flags}
"""),
    md("## Comprobación sin simulador (episodios que la red no vio)"),
    writefile("offline_eval_policy.py", ROOT / "tools" / "offline_eval_policy.py"),
    code("""
!python offline_eval_policy.py --policy "{OUT}/checkpoints/last/pretrained_model" --dataset-root "{DATA}" --episodes 450-499 --stride 5 --out "{OUT}/offline_eval.json"
"""),
    md("""
Cómo leerlo: el error de la política a los pasos 25 y 49 del bloque tiene que ser **claramente menor** que el de «copiar el estado». Si no, la red no ha
aprendido a anticipar: más pasos, más episodios o revisar el dataset (`tools/validate_dataset.py`). Si es menor, el siguiente juez es Isaac.
"""),
    code("""
import shutil, json
shutil.make_archive("/content/pretrained_model", "zip", f"{OUT}/checkpoints/last/pretrained_model")
shutil.copy("/content/pretrained_model.zip", f"/content/drive/MyDrive/piezas/act_{OBS_MODE}.zip")
print(json.load(open(f"{OUT}/checkpoints/last/pretrained_model/binpick_info.json")))
from google.colab import files; files.download("/content/pretrained_model.zip")
"""),
    md("""
## Usarlo en el demo

```powershell
# descomprime en demos/09_voice_bin_picking/outputs/act_rgb_mask/pretrained_model
python tools/eval_sorting.py --headless --seeds 20 --policy act:outputs/act_rgb_mask/pretrained_model     # el juez real
python 18_voice_bin_picking.py --policy act+guion:outputs/act_rgb_mask/pretrained_model --voice --open     # demo con respaldo del guion (se etiqueta)
```

## Más datos / afinado

Si la política falla en situaciones concretas (pilas, piezas cerca de la pared), graba **más** episodios de ese tipo
(`collect_demos.py --resume --layout apiladas --first-seed 5000`), sube el dataset ampliado y continúa desde el checkpoint:

```
!python train_act.py --dataset-root "{DATA}" --obs-mode {OBS_MODE} --steps 20000 --init-from "{OUT}/checkpoints/last/pretrained_model" --output /content/outputs/act_v2
```
"""),
])


def main() -> int:
    out = ROOT / "colab"
    out.mkdir(exist_ok=True)
    for name, nb in (("01_entrenar_yolo_seg.ipynb", YOLO), ("02_entrenar_act.ipynb", ACT)):
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("escrito", out / name, f"({len(nb['cells'])} celdas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
