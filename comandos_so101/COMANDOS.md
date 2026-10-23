# Comandos SO-101: calibración y movimiento

Comandos usados en este equipo para los dos brazos:

| Brazo | Rol | Puerto | ID LeRobot |
|---|---|---|---|
| Teacher | Leader / teleoperador | `/dev/ttyACM0` | `mi_leader` |
| Student | Follower / robot | `/dev/ttyACM1` | `mi_follower` |

## Preparar el terminal

Desde la raíz del repositorio, activa el entorno virtual:

```bash
cd ~/Workspace/Robotekin_
source .venv/bin/activate
```

El usuario debe pertenecer al grupo `dialout` para acceder a los puertos serie:

```bash
id -nG
```

Si `dialout` no aparece, añádelo y cierra la sesión de Ubuntu y vuelve a iniciarla:

```bash
sudo usermod -aG dialout "$USER"
```

Instala las dependencias opcionales de Feetech y calibración si falta alguna:

```bash
uv pip install 'lerobot[feetech,deepdiff-dep]'
```

## Calibrar el follower

Conecta la fuente de alimentación del follower, despeja la zona de movimiento y ejecuta:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=mi_follower
```

Coloca cada articulación aproximadamente en el centro de su recorrido cuando lo pida el programa. Después mueve cada articulación despacio por todo su rango, incluida la pinza; `wrist_roll` no requiere recorrido. No fuerces los topes.

Si pregunta si quieres reutilizar una calibración guardada, pulsa `c` y luego Enter para hacer una nueva.

## Calibrar el teacher / leader

Ejecuta:

```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=mi_leader
```

Sigue las instrucciones de centrado y recorrido completo. Si aparece la pregunta sobre reutilizar la calibración anterior, pulsa `c` y luego Enter para recalibrar.

> Ejecuta `lerobot-setup-motors` solo si los IDs de motor no están configurados. No lo ejecutes en brazos ya preparados; consulta primero a la persona responsable del hardware.

## Teleoperar: mover el follower con el leader

Con ambos brazos conectados y calibrados, ejecuta:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=mi_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=mi_leader
```

Mueve el leader lentamente y comprueba que el follower lo imita. Pulsa `Ctrl+C` para detener el programa. Si el follower se mueve de forma brusca o en sentido inesperado, pulsa `Ctrl+C` y revisa la asignación de puertos y las calibraciones antes de volver a probar.
