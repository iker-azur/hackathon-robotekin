# Preparar el PC para grabar el SO-101

Esta guía reduce cortes y retraso al grabar con dos webcams a 1280 × 720. Los
cambios de energía son temporales; no cambies límites de corriente, torque ni
protecciones del brazo.

## 1. Ver los modos de las cámaras y la conexión USB

Instala las herramientas si faltan:

```bash
sudo apt update
sudo apt install v4l-utils usbutils linux-cpupower
```

Comprueba qué resoluciones, formatos y FPS ofrece cada cámara:

```bash
v4l2-ctl --device=/dev/video2 --list-formats-ext
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

Comprueba en qué controlador USB está conectada cada cámara:

```bash
lsusb -t
```

En este PC ambas cámaras aparecen en el bus raíz USB 2.0 (480 Mbit/s), una tras
un hub. Conecta cada webcam directamente al PC y prueba puertos que aparezcan
bajo buses/controladores distintos en `lsusb -t`; evita hubs compartidos. Vuelve
a ejecutar el comando para comprobar el cambio. Las cámaras pueden seguir
mostrando `480M`: eso es la velocidad negociada por la webcam, aunque el puerto
del PC sea USB 3.x.

Si el modo 1280 × 720 está disponible como **MJPG**, es preferible a YUYV para
reducir el tráfico USB. Solo si ambas cámaras listan ese modo, añade
`fourcc="MJPG"` a las dos instancias `OpenCVCameraConfig` de
`17_record_real_so101.py`; el script todavía no ofrece un argumento CLI para
seleccionar FOURCC. Para grabar a esa resolución, mantén
`--camera-width 1280 --camera-height 720`; usa `--fps 10`, que es el valor
predeterminado actual y el que se verificó con estas cámaras.

## 2. Activar temporalmente el perfil de rendimiento

Comprueba los perfiles disponibles y activa rendimiento antes de grabar:

```bash
powerprofilesctl list
powerprofilesctl set performance
powerprofilesctl get
```

Al acabar, vuelve al perfil equilibrado:

```bash
powerprofilesctl set balanced
```

Si `powerprofilesctl` no ofrece el perfil `performance`, no lo fuerces. Como
alternativa, comprueba primero el gobernador disponible y actívalo
temporalmente:

```bash
cpupower frequency-info -g
sudo cpupower frequency-set -g performance
```

El cambio de `cpupower` no suele persistir tras reiniciar. Para restaurar el
gobernador, usa el nombre que mostraba el primer comando, por ejemplo:

```bash
sudo cpupower frequency-set -g schedutil
```

## 3. Comprobar carga mientras grabas

En otra terminal, observa CPU y procesos:

```bash
htop
```

Observa errores de cámara/USB del kernel:

```bash
sudo dmesg --follow
```

Cierra visores de vídeo y programas que estén usando las webcams antes de
iniciar el recorder. Si hay retraso, prueba primero mover las cámaras a
controladores USB distintos. Como diagnóstico, compara una grabación a
640 × 480 con otra a 1280 × 720; no cambies el FPS mientras comparas. Si solo
la resolución alta presenta retraso, limita el tráfico USB con MJPG si está
disponible o reduce la resolución. Si el uso de CPU se mantiene al máximo,
cierra procesos pesados y comprueba de nuevo.

Inicia la grabación desde el entorno del proyecto:

```bash
source .venv/bin/activate
python demos/06_imitation_learning/17_record_real_so101.py
```
