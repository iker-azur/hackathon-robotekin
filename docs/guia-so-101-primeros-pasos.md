# SO-101: primera conexión y primeros movimientos

Guía mínima para el hackathon. Todo se ejecuta en **PowerShell**, desde la carpeta del repo:

```powershell
cd C:\Users\Iker\Documents\Workspace\HackathonRobotekin_
```

El entorno ya está instalado en `lerobot-env` (LeRobot 0.6.1 + SDK de Feetech).

---

## 0. Antes de conectar (30 segundos)

- Pregunta a la organización: **¿los IDs de los motores ya están configurados?** Si el brazo ya viene montado y probado, la respuesta será sí y te saltas el paso 3.
- Conecta **la fuente de alimentación del follower**. El USB solo no mueve sus motores; el leader no lleva fuente.
- Cada brazo va por su propio USB y aparece como un **puerto COM distinto**. No los confundas: leader y follower no son intercambiables.
- Despeja la mesa. El follower se pone rígido en cuanto tiene torque.

---

## 1. Activar el entorno (una vez por terminal)

```powershell
.\lerobot-env\Scripts\Activate.ps1
```

Si PowerShell lo bloquea:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\lerobot-env\Scripts\Activate.ps1
```

Sabrás que funciona porque el prompt empieza por `(lerobot-env)` y existe el comando `lerobot-calibrate`.

---

## 2. Averiguar qué COM es cada brazo

Conecta **solo el leader** por USB y ejecuta:

```powershell
lerobot-find-port
```

1. Te lista los puertos que ve.
2. Te pide desconectar el USB del brazo. Desconéctalo y pulsa **Enter**.
3. Te dice qué puerto era. Vuelve a conectarlo.

Repite lo mismo con **solo el follower** conectado. Anota los dos y guárdalos en variables:

```powershell
$LEADER   = "COM4"   # cambia por el que te haya dicho
$FOLLOWER = "COM5"   # cambia por el que te haya dicho
```

Si no quieres el script: abre el Administrador de dispositivos, desenchufa y enchufa un brazo y mira qué COM aparece y desaparece.

> **Truco:** si `lerobot-find-port` dice `Could not detect the port`, es casi siempre porque el cable no estaba bien puesto o porque otro programa (Arduino IDE, monitor serie, etc.) tiene el COM abierto. Cierra eso y repite.

---

## 3. Configurar los motores (solo si el brazo es nuevo)

**No lo hagas en un brazo ya montado y configurado**: hay que desconectar los motores uno a uno y se pueden dejar los IDs mal. Pregunta antes.

El orden es: `gripper` → `wrist_roll` → `wrist_flex` → `elbow_flex` → `shoulder_lift` → `shoulder_pan`. El script te va pidiendo conectar **un único motor** a la placa controladora cada vez.

Follower:

```powershell
lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER
```

Leader:

```powershell
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=$LEADER
```

Cuando termine, vuelve a conectar la cadena de cables normal (motor 1 → motor 2 → ... → placa).

---

## 4. Calibrar el follower (el brazo con motores)

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=mi_follower
```

Qué te va a pedir:

1. **`Move ... to the middle of its range of motion and press ENTER`**
   Coloca el brazo en una postura cómoda a mitad de recorrido: base centrada, hombro ni del todo arriba ni abajo, codo a unos 90°, muñeca recta, pinza medio abierta. Pulsa **Enter**.
2. **`Move all joints except 'wrist_roll' sequentially through their entire ranges of motion`**
   Coge cada articulación con la mano y llévala despacio hasta sus dos topes (ida y vuelta). Incluye abrir y cerrar la pinza del todo. `wrist_roll` no hace falta: gira sin límite. Pulsa **Enter** al acabar.
3. Imprime `Calibration saved to ...` y desconecta.

Tarda 5–10 minutos. Si te equivocas, se repite: no pasa nada.

> **Ojo con el id.** La calibración se guarda asociada a `--robot.id`. Si ya existe un archivo con ese id, el comando te preguntará:
> `Press ENTER to use provided calibration file ... or type 'c' and press ENTER to run calibration`
> Pulsa **Enter** solo si es el mismo brazo físico. Si es otro robot o no te fías, escribe **`c`** + Enter para recalibrar.
> Si cambias de brazo, cambia el id (por ejemplo `mi_follower_2`).

---

## 5. Calibrar el leader (el brazo que mueves a mano)

```powershell
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=mi_leader
```

Mismos dos pasos. El leader no tiene motores activos, así que lo mueves a mano sin resistencia. La pinza se abre y cierra con el gatillo.

---

## 6. Mover el robot

```powershell
lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=mi_follower --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=mi_leader
```

- Mueve el **leader** con la mano y el **follower** te copia.
- Empieza despacio y con recorridos cortos.
- **Ctrl+C** para parar.
- Si el follower se mueve al revés, a saltos, o con una articulación loca: puertos intercambiados o calibración mal hecha. Para, revisa y recalibra.

Con esto ya tienes el robot conectado y teleoperado. Es el 80% del trabajo del primer día.

---

## 7. Siguiente paso: grabar demostraciones

> **Aviso:** en este portátil `lerobot-record` todavía no funciona (ver la nota al final de esta guía). Los comandos de esta sección son los correctos, pero hay que arreglar `transformers` antes de ejecutarlos.

Antes de grabar, comprueba las cámaras:

```powershell
lerobot-find-cameras
```

Graba 5 episodios de prueba (sin subir a Hugging Face):

```powershell
lerobot-record --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=mi_follower --robot.cameras='{ frontal: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=mi_leader --dataset.repo_id=mi_usuario/so101_prueba --dataset.num_episodes=5 --dataset.single_task="coger el bloque" --dataset.push_to_hub=false --dataset.streaming_encoding=true
```

Teclas durante la grabación: **→** o **n** siguiente episodio, **←** o **r** repetir, **Esc** o **q** terminar.

---

## 8. Problemas habituales

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `Could not detect the port` | Cable flojo u otro programa usando el COM | Revisa cable, cierra Arduino IDE / monitores serie, repite |
| `PermissionError` o `Acceso denegado` al COM | Otro programa tiene el puerto abierto | Cierra ese programa y reconecta el USB |
| El follower no se mueve | Sin fuente de alimentación, sin calibrar o COM equivocado | Revisa alimentación y que `$FOLLOWER` sea el brazo con motores |
| Se mueve invertido o hace cosas raras | Leader y follower intercambiados, o calibración de otro robot | Cambia los COM y recalibra con un id nuevo |
| `Incorrect status packet` | Cable de 3 hilos flojo en la cadena | Revisa las conexiones entre motores y reintenta |
| Una articulación no responde | Cable de esa articulación suelto | Revisa el cable de ese motor en la cadena |
| El brazo se queda rígido y no puedo moverlo | Tiene torque activo | Es normal: usa el leader o Ctrl+C, no force el follower |

---

## Nota sobre este portátil

`lerobot-calibrate` y `lerobot-teleoperate` funcionan con el entorno actual. **`lerobot-record` y `lerobot-train` no**: el venv tiene `transformers 4.57.6` y LeRobot 0.6.1 importa las políticas esperando `transformers >= 5.4, < 5.6`. Falla con:

```
ImportError: cannot import name 'torch_compilable_check' from 'transformers.utils'
```

Antes de grabar/entrenar hay que actualizar `transformers` en `lerobot-env`:

```powershell
uv pip install --python .\lerobot-env\Scripts\python.exe "transformers>=5.4,<5.6"
```

Para volver atrás si algo se rompe:

```powershell
uv pip install --python .\lerobot-env\Scripts\python.exe "transformers==4.57.6"
```

No lo he ejecutado en este entorno: los demos de percepción pueden depender de la versión actual. Si quieres, lo pruebo y verifico que grabar y entrenar funcionan antes de tocar nada.

Además, el `torch` instalado es **CPU only**. Para entrenar ACT con 50 episodios, usa una máquina con GPU, Colab o HF Jobs; en este portátil tardaría muchísimo.
