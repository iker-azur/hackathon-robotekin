# SO-101 Autonomous Agent

## Guía conceptual y práctica para Robotekin Hackathon LeRobot 2026

**Investigación revisada el 23 de septiembre de 2026.**

> El objetivo de esta guía es comprender las ideas y construir un buen prototipo. No es una guía de instalación ni de APIs.

---

## Respuesta breve

Con un SO-101 y LeRobot podemos construir un robot que:

1. sea controlado manualmente;
2. pueda funcionar mediante teleoperación;
3. registre una tarea demostrada por una persona;
4. aprenda esa tarea a partir de ejemplos;
5. actúe solo;
6. se adapte a nuevas posiciones;
7. compruebe si ha tenido éxito;
8. y quizá intente corregirse.

La progresión recomendada es:

```text
Controlar
   ↓
Demostrar
   ↓
Registrar
   ↓
Aprender
   ↓
Evaluar
   ↓
Añadir visión
   ↓
Añadir autonomía
   ↓
Verificar y recuperarse
```

La idea central de nuestro proyecto debe entenderse como una **evolución progresiva**:

```text
VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE
```

No como un agente complejo que debamos construir entero desde el primer día.

---

# 1. El hackathon Robotekin

## Qué es Robotekin

[Robotekin](https://robotekin.com/robotekin/) es la Asociación Vasca de Robótica y Automatización. Conecta empresas, universidades, centros tecnológicos, centros de formación, startups y otras organizaciones interesadas en robótica.

El [Robotekin Hackathon LeRobot 2026](https://hackathon.robotekin.com/) está anunciado para los días **23, 24 y 25 de octubre de 2026** en Vitoria-Gasteiz.

El evento busca crear prototipos funcionales que combinen:

- robótica real;
- datos obtenidos del mundo físico;
- aprendizaje por imitación;
- visión;
- IA aplicada;
- trabajo en equipo durante aproximadamente 48 horas.

## Hardware anunciado

La FAQ oficial indica que habrá:

- **10 kits SO-101**;
- ordenadores;
- cámaras;
- trípodes.

La distribución exacta de los equipos, los modelos de cámara y los objetos disponibles no están publicados. La web también muestra cifras discrepantes de aforo —75 participantes en una sección y 50 en la FAQ—, por lo que esos detalles deben confirmarse con la organización.

## El papel de LeRobot

LeRobot no es el robot ni el track. Es el **framework que conecta**:

- hardware;
- teleoperación;
- grabación de datasets;
- entrenamiento de policies;
- inferencia;
- Hugging Face Hub.

Su flujo conceptual es:

```text
Teleoperar → Registrar → Entrenar → Desplegar
```

## Tracks oficiales

La [web oficial](https://hackathon.robotekin.com/) publica cuatro áreas:

1. **Manipulación de objetos:** agarrar, mover, clasificar o colocar objetos.
2. **Visión robótica:** interpretar una escena mediante cámaras.
3. **Aprendizaje por imitación:** aprender de demostraciones humanas.
4. **Aplicaciones sociales:** educación, accesibilidad, asistencia o impacto social.

Se puede participar en una o varias áreas.

## Cómo se combinan los tracks

Un mismo proyecto puede cubrir manipulación, visión e imitación:

```text
Cámara
  ↓
El robot observa un cubo rojo
  ↓
La visión decide qué objeto manipular
  ↓
Una policy aprende a agarrar y colocar
  ↓
La cámara comprueba el resultado
```

También puede incorporar una aplicación social:

```text
"Pásame el bloque rojo"
  ↓
El robot lo coloca en una bandeja accesible
```

No hace falta demostrar cuatro proyectos independientes. Es más sólido mostrar una tarea clara que combine varias capacidades.

---

# 2. Qué es la robótica moderna

La robótica moderna combina hardware, percepción, control, software y aprendizaje.

Un robot no recibe una orden y simplemente la ejecuta. Normalmente trabaja en un ciclo:

```text
Observar
  ↓
Decidir
  ↓
Actuar
  ↓
Comprobar el efecto
  ↓
Observar de nuevo
```

## Observación

El robot recibe información mediante:

- cámaras;
- posiciones de sus articulaciones;
- estado del gripper;
- sensores de fuerza o distancia, si los tiene;
- información de la tarea.

## Decisión

Algún sistema decide qué movimiento realizar. Puede ser:

- una regla programada;
- un algoritmo de control;
- una policy entrenada;
- un sistema que combina varias técnicas.

## Acción

El robot modifica el mundo. Por ejemplo:

- mueve una articulación;
- abre el gripper;
- acerca el brazo a un cubo;
- suelta un objeto.

## Resultado

La acción puede tener efectos diferentes de los previstos:

- el cubo resbala;
- el gripper no lo sujeta;
- una persona mueve el objeto;
- la cámara queda tapada;
- el robot golpea la caja.

La robótica es difícil porque el robot **cambia el entorno que está observando**.

## Relación con nuestro proyecto

No tenemos que intentar resolver toda la robótica moderna. Tenemos que construir un sistema pequeño que cierre bien un ciclo concreto:

```text
Ver una escena
  ↓
Realizar una tarea
  ↓
Comprobar qué ocurrió
```

---

# 3. Qué es realmente un SO-101

El [SO-101](https://huggingface.co/docs/lerobot/so101) es un brazo robótico abierto y de bajo coste pensado principalmente para experimentar con teleoperación e imitation learning.

## Qué tipo de robot es

Es un brazo de sobremesa de aproximadamente cinco grados de libertad, más un gripper:

```text
Base giratoria
   ↓
Hombro
   ↓
Codo
   ↓
Muñeca
   ↓
Gripper
```

En la práctica, LeRobot maneja seis canales de movimiento:

1. rotación de la base;
2. elevación del hombro;
3. flexión del codo;
4. flexión de la muñeca;
5. giro de la muñeca;
6. apertura y cierre del gripper.

## Partes principales

Sin entrar en electrónica, un SO-101 contiene aproximadamente:

- base;
- estructura del hombro;
- brazo superior;
- antebrazo;
- muñeca;
- gripper o pinza;
- motores y controlador;
- cables;
- estructura impresa en 3D;
- fuentes de alimentación;
- leader adicional para teleoperación.

## Qué es el gripper

El **gripper**, también llamado end effector o efector final, es la parte que entra en contacto con el objeto.

En una pinza sencilla:

```text
Abrir  → intentar coger
Cerrar → sujetar
Abrir  → soltar
```

Cerrar el gripper no demuestra que el objeto esté correctamente agarrado. Puede estar:

- demasiado cerca de la pinza;
- torcido;
- fuera de los dedos;
- demasiado apretado;
- sin sujetar realmente el objeto.

## Qué tareas encajan

Es apropiado para:

- coger un cubo;
- mover una pieza pequeña;
- colocar un objeto en una caja;
- clasificar objetos;
- escoger uno según su color;
- apilar dos piezas;
- limpiar una mesa pequeña;
- insertar una pieza con tolerancias amplias;
- entregar un objeto a una persona dejándolo en una bandeja.

## Qué no debemos esperar

No debemos esperar que sea:

- un robot industrial certificado;
- rápido como un robot de almacén;
- preciso para piezas de milímetros;
- capaz de levantar objetos pesados;
- seguro para contacto humano;
- able para moverse por una habitación;
- un manipulador universal;
- un humanoid capaz de tomar decisiones.

Las fuentes oficiales no publican una ficha industrial completa de alcance, carga útil, velocidad o repetibilidad. Por eso debemos probar esos límites con objetos pequeños y tareas controladas.

## SO-100 y SO-101

El SO-101 es la evolución del SO-100. Mejora aspects como cableado, montaje y motores del leader. Ambos son muy parecidos y comparten mucho soporte, pero no debemos asumir que todas sus piezas, calibraciones, modelos o policies son intercambiables.

## Relación con nuestro proyecto

El SO-101 debe ser el **actor físico** del proyecto. LeRobot será la capa que nos permitirá controlarlo, recoger datos y aprender policies.

---

# 4. Leader y Follower

Este es uno de los conceptos centrales de LeRobot.

## Leader

El leader es un brazo que funciona como mando físico.

## Follower

El follower es el robot que ejecuta los movimientos.

```text
Humano mueve el SO-101 Leader
            ↓
LeRobot lee las posiciones
            ↓
El SO-101 Follower copia esos movimientos
```

En términos sencillos:

> Un brazo sirve como mando físico y el otro copia sus movimientos.

## Por qué es interesante

La teleoperación permite:

- controlar el robot de forma inmediata;
- ver qué movimientos son necesarios;
- grabar demostraciones;
- corregir al robot en tiempo real;
- aprender habilidades que todavía no sabemos programar.

## Calibración

Leader y follower deben compartir una representación coherente de sus ángulos. La calibración determina:

- la posición central de cada articulación;
- su rango;
- la correspondencia entre leader y follower.

Una calibración incorrecta puede hacer que ambos brazos parezcan iguales para una persona, pero produzcan valores diferentes.

## De la teleoperación al aprendizaje

```text
Humano mueve el leader
       ↓
Follower realiza el movimiento
       ↓
Las cámaras observan
       ↓
Se registran imágenes, estados y acciones
       ↓
Tenemos una demostración
```

La teleoperación no siempre necesita grabar. Es útil por sí misma para:

- comprobar el hardware;
- entender los límites del robot;
- practicar;
- grabar futuros datasets de policies.

---

# 5. Cómo puede aprender o controlar un robot

Existen varias formas de conseguir que un robot realice una tarea.

## 5.1 Programación tradicional

El programador define una secuencia:

```text
Mover el brazo hasta A
  ↓
Abrir el gripper
  ↓
Bajar
  ↓
Cerrar el gripper
  ↓
Subir
  ↓
Mover hacia la caja
  ↓
Abrir el gripper
```

### Ventajas

- Es predecible.
- Es fácil de verificar.
- Puede ser suficiente para una secuencia fija.
- No necesita datos.

### Limitaciones

- Hay que escribir cada movimiento.
- Resulta difícil responder a cambios.
- No generaliza bien a nuevas posiciones.
- Se complica cuando hay varios objetos o errores.

Ejemplo: el robot puede seguir moviéndose aunque la caja haya cambiado de lugar.

## 5.2 Teleoperación

Una persona controla directamente al robot.

```text
Persona → Robot
```

### Ventajas

- Permite realizar la tarea sin programación compleja.
- Genera datos de acciones.
- Es la base habitual del imitation learning.

### Limitaciones

- Requiere una persona.
- No es autonomía.
- Solo demuestra lo que la persona haga.

## 5.3 Imitation Learning

La persona realiza la tarea muchas veces. Después, un modelo aprende a relacionar:

```text
Lo que el robot observa
        +
Lo que el robot está haciendo
        ↓
La acción que debería realizar
```

### Analogía

Imagina que un niño aprende a hacer un nudo:

1. ve a otra haciéndolo;
2. observa los movimientos;
3. intenta imitar;
4. recibe correcciones;
5. mejora con práctica.

El robot hace lo mismo con datos.

### Behavior Cloning

La forma más sencilla de imitation learning se llama **behavior cloning**:

```text
Ejemplos humanos
  ↓
Modelo que aprende la relación
observación → acción
```

Su ventaja es su simplicidad. Su problema es que puede imitar también los errores contenidos en los datos.

### ACT

[ACT](https://huggingface.co/docs/lerobot/act), Action Chunking Transformer, es una policy adecuada para empezar:

- relativamente pequeña;
- rápida de entrenar;
- utiliza imágenes y estado;
- predice secuencias de acciones;
- no necesita un modelo de lenguaje.

## 5.4 Reinforcement Learning

En reinforcement learning, el robot prueba acciones y recibe una recompensa:

```text
Intentar
  ↓
Conseguir resultado útil → recompensa positiva
Tomar una mala decisión → recompensa negativa
  ↓
Aprender qué acciones funcionan mejor
```

Es interesante cuando se puede definir y medir bien una recompensa, pero resulta costoso en robótica real porque cada intento puede:

- tardar minutos;
- gastar datos;
- romper un objeto;
- dañar el equipo;
- requerir una persona que lo reinicie.

Por eso no es un punto de partida natural para nuestro hackathon.

## 5.5 Vision-Language-Action

Un modelo VLA recibe:

```text
Imágenes
  +
Instrucción en lenguaje
  +
Estado del robot
  ↓
Acciones del robot
```

Ejemplo:

```text
"Coge el cubo rojo y ponlo en la caja"
  +
Cámara
  +
Posición del brazo
  ↓
VLA
  ↓
Secuencia de movimientos
```

### Por qué es interesante

Un VLA puede intentar combinar:

- qué se ve;
- qué se pide;
- qué está haciendo el robot.

### Por qué no deberíamos empezar aquí

Un VLA añade:

- complejidad;
- más datos;
- entrenamiento más exigente;
- latencia;
- más modos de fallo;
- necesidad de verificar si sigue la instrucción.

[SmolVLA](https://huggingface.co/docs/lerobot/smolvla) es un ejemplo compacto y abierto, pero el checkpoint base no sustituye la grabación y el ajuste con datos propios.

Nuestra progresión debería ser:

```text
Policy visual sencilla
  ↓
Policy condicionada por lenguaje
  ↓
VLA
```

---

# 6. Qué es una demostración

Una demostración es un intento de completar una tarea cuyo comportamiento queda registrado.

Tarea:

> "Coge el cubo rojo y colócalo en la caja."

## Ejemplo

### Momento 1

La cámara ve la mesa. El robot está quieto.

### Momento 2

El gripper se acerca al cubo.

### Momento 3

El gripper se abre.

### Momento 4

El brazo baja hasta una posición adecuada.

### Momento 5

El gripper se cierra.

### Momento 6

El robot levanta el cubo.

### Momento 7

El robot mueve el cubo hacia la caja.

### Momento 8

El gripper se abre y el cubo cae dentro.

## Qué se registra

Durante toda la demostración pueden almacenarse:

- imágenes de cámaras;
- posiciones de las articulaciones;
- estado del gripper;
- acciones enviadas;
- instante de tiempo;
- número de episodio;
- número de fotograma;
- descripción de la tarea;
- estado final;
- si tuvo éxito.

## Episodio

Un **episodio** es un intento completo de realizar la tarea.

Un episodio puede ser:

- exitoso;
- fallido;
- parcialmente exitoso;
- corregido por una persona.

No debemos eliminar automáticamente todos los intentos fallidos. Primero hay que registrarlos y entender qué ocurrió.

---

# 7. Qué es un dataset robótico

> Un dataset robótico es una colección de experiencias.

Cada experiencia contiene observaciones, acciones y contexto.

## Estructura conceptual

```text
Dataset
├── Episodio 1
│   ├── Imágenes
│   ├── Estado
│   ├── Acciones
│   └── Tarea
├── Episodio 2
├── Episodio 3
└── ...
```

## Un ejemplo

Podemos tener 50 episodios de:

> "Coger el cubo rojo y colocarlo en la caja."

## Calidad de las demostraciones

No todos los ejemplos valen lo mismo.

### Demostraciones buenas

- se acerca al objeto correctamente;
- sujeta el objeto;
- los movimientos son fluidos;
- no hay movimientos innecesarios;
- la suelta es limpia;
- se puede imitar fácilmente.

### Demostraciones malas

- el humano mira directamente al robot y no a la cámara;
- el objeto está fuera de la distribución;
- el grasp es pobre;
- el robot roza la mesa;
- el humano corrige movimientos inexperienced;
- se pierde el objeto;
- hay pausas largas.

Una demostración mala no siempre debe eliminarse. Puede ser útil si la sabemos identificar y utilizar para entrenar recuperación o clasificación de finales.

## Variedad

Un buen dataset puede incluir:

- distintas posiciones;
- pequeñas rotaciones;
- varios objetos similares;
- distintas iluminaciones;
- diferentes fondos;
- distractores;
- varios ángulos de aproximación;
- objetos parcialmente tapados.

## Consistencia

Al principio también necesitamos consistencia:

- mismas cámaras;
- mismo encuadre;
- mismos objetos;
- mismo criterio de éxito;
- comportamiento de grasp razonablemente homogéneo.

El equilibrio es:

```text
Primero: pocas demostraciones consistentes
Después: más variedad controlada
```

## Cantidad

La documentación de LeRobot sugiere alrededor de 50 episodios como punto de partida para una tarea sencilla, repartidos entre varias posiciones.

Eso no es una cantidad mínima universal. Si el objeto es heterogéneo o la tarea tiene muchas variantes, más episodios pueden no ser suficientes.

## Calidad frente a cantidad

Cincuenta ejemplos pueden ser mejores que 200 si:

- están bien ejecutados;
- cubren la tarea correctamente;
- tienen una definición clara;
- no contienen movimientos ambiguos;
- el criterio de éxito es verificable.

Recoger más datos sin analizar los errores puede hacer que la policy procese peor la tarea.

---

# 8. Datasets reales de la comunidad

Los números siguientes proceden de los manifests y tarjetas consultables. Algunos papers describen una versión anterior con 50 episodios, mientras que el manifest actual contiene 52 o 56; se indica la diferencia.

## 8.1 SO-100 Pick and Place

[Dataset: `lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace)

- **Robot:** SO-100.
- **Tarea:** coger un cubo y colocarlo en una caja.
- **Datos:** estado y acciones articulares de seis dimensiones.
- **Episodios:** 50.
- **Fotogramas:** 19.631.
- **Cámaras:** superior y muñeca, 480×640, 30 fps.
- **Variación:** el paper de SmolVLA describe cinco posiciones iniciales y diez episodios por posición.

**Qué aprendemos:** es un referencia muy cercano a una primera tarea real. Permite estudiar grasp, lift, transport y place con un dataset relativamente pequeño.

## 8.2 SO-100 Sorting

[Dataset: `lerobot/svla_so100_sorting`](https://huggingface.co/datasets/lerobot/svla_so100_sorting)

- **Robot:** SO-100.
- **Tarea:** colocar el cubo rojo en la caja derecha y el azul en la izquierda.
- **Episodios:** 52 en el manifest actual; 50 en la versión descrita por el paper.
- **Fotogramas:** 35.713.
- **Cámaras:** superior y muñeca, 480×640, 30 fps.
- **Variación:** el paper describe cinco distribuciones o layouts.

**Qué aprendemos:** sorting obliga a mantener el estado de la escena. No basta con una policy que simplemente coja algo: debe saber qué objeto y qué destino.

## 8.3 SO-100 Stacking

[Dataset: `lerobot/svla_so100_stacking`](https://huggingface.co/datasets/lerobot/svla_so100_stacking)

- **Robot:** SO-100.
- **Tarea:** colocar el cubo rojo encima del azul.
- **Episodios:** 56 actualmente; 50 en el paper.
- **Fotogramas:** 22.956.
- **Cámaras:** superior y muñeca, 480×640, 30 fps.
- **Variación:** las posiciones iniciales de los cubos cambian.

**Qué aprendemos:** colocar exige levantar, mantener el objeto, alinearlo, bajarlo y soltarlo en el momento correcto.

## 8.4 SO-101 Pick and Place

[Dataset: `lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)

- **Robot:** el paper lo identifica como SO-101.
- **Tarea:** colocar una pieza Lego rosa en una caja transparente.
- **Episodios:** 50.
- **Fotogramas:** 11.939.
- **Cámaras:** `up` y `side`, 480×640, 30 fps.
- **Variación:** el paper describe cinco posiciones iniciales con diez episodios por posición.

**Advertencia:** el manifest actual declara `robot_type: so100_follower`, aunque el paper y el nombre del dataset lo describen como SO-101. No debemos asumir que un checkpoint o un archivo de calibración de otro embodiment funciona sin verificarlo.

## 8.5 SO-101 Teleoperated Cubes

[Dataset: `qb1t/so101_teleop_cubes`](https://huggingface.co/datasets/qb1t/so101_teleop_cubes)

- **Robot:** SO-101 follower.
- **Tarea:** coger cubos rojos y amarillos y colocarlos en un cuenco blanco.
- **Episodios:** 50.
- **Fotogramas:** 38.777.
- **Cámaras:** muñeca (`ego`) y frontal (`external_D455`), 480×640, 30 fps.
- **Posiciones y distractores:** no están documentados con detalle.

**Qué aprendemos:** la cámara frontal ayuda a identificar colores y el objetivo; la cámara de muñeca ayuda durante el contacto. También hay que mantener cuántas piezas ya se han colocado.

## 8.6 Sponge with Distractors

[Dataset: Sponge con distractores](https://huggingface.co/datasets/aswinkumar99/LeRobot-SO101-task3-single-sponge-with-distractors-random-locations)

- **Robot:** SO-101, según la tarjeta comunitaria.
- **Tarea:** coger una esponja azul y colocarla en un cuenco.
- **Episodios:** 64.
- **Fotogramas:** 34.725.
- **Cámaras:** overhead y muñeca, 480×640, 30 fps.
- **Variación:** layouts aleatorios, distractores e iluminación de mañana y noche.

**Qué aprendemos:** la robustez no depende solo del color. También importan oclusiones, orientación, iluminación y presencia de otros objetos.

## 8.7 Naive Bench

[Dataset: `binhpham/naive-bench`](https://huggingface.co/datasets/binhpham/naive-bench)

- **Robot:** SO-101.
- **Tarea:** "coloca la barra de color X en la bandeja de color Y".
- **Episodios:** 312.
- **Fotogramas:** 115.378.
- **Cámaras:** brazo y overhead, 240×320, 30 fps.
- **Lenguaje:** 24 combinaciones posibles; 20 aparecen en entrenamiento y cuatro se reservan para evaluar composición.

**Qué aprendemos:** una policy puede memorizar colores o destinos. Reservar combinaciones completas permite saber si realmente entiende ambos elementos de la instrucción.

## 8.8 Autonomous Pick and Place

[Dataset: `legalaspro/so101_pickplace_auto_v1`](https://huggingface.co/datasets/legalaspro/so101_pickplace_auto_v1)

- **Robot:** SO-101 físico.
- **Tarea:** coger un cubo verde y colocarlo en una zona objetivo.
- **Episodios:** 200.
- **Fotogramas:** 129.530.
- **Cámaras:** wrist y top, 480×640, 50 fps.
- **Origen:** episodios autónomos, no demostraciones humanas teleoperadas.
- **Etiquetas de éxito y distribución de posiciones:** no documentadas.

**Qué aprendemos:** también podemos recoger datos cuando una policy intenta la tarea. Pero sus errores no deben alimentarse sin filtrar. Es útil para datos correctivos y para estudiar modos de fallo.

## 8.9 ArmnetBench

[Dataset: `armnet/armnetbench_v01_lerobot_so101`](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101)

- **Robot:** SO-101 de un brazo.
- **Tareas:** ocho.
- **Demostraciones humanas:** 50 por tarea.
- **Evaluación:** episodios producidos por siete policies.
- **Episodios totales:** 2.499.
- **Fotogramas:** 1.127.881.
- **Cámaras:** front, top y wrist.
- **Etiquetas:** successful, failure y suboptimal.

**Qué aprendemos:** el conjunto no solo guarda movimientos. También permite estudiar calidad, fallos y comparabilidad. Es especialmente útil para pensar en verificación y recuperación.

---

# 9. Qué significa entrenar una policy

## Definición

Una **policy** es el sistema que decide qué acción debe realizar el robot a partir de lo que observa.

No tiene por qué ser una red neuronal. Una regla también puede ser una policy.

## Flujo

```text
Observación
   ↓
Policy
   ↓
Acción
```

Ejemplo:

```text
Observación:
- El cubo está delante.
- El brazo está a la izquierda.
- El gripper está abierto.

Policy:
- Mover base un poco a la derecha.
- Bajar el hombro.
- Cerrar cuando esté centrado.
```

## Action chunking

Algunas policies no producen una única acción, sino varias:

```text
Acción actual
Siguiente acción
Después
Más tarde
```

Esto permite producir movimientos más fluidos.

## Tipos de policy

- **Regla:** si el cubo está a la izquierda, gira.
- **Behavior cloning:** imita ejemplos.
- **ACT:** predice action chunks.
- **Diffusion Policy:** genera secuencias mediante un proceso de refinamiento.
- **VLA:** usa visión, lenguaje y estado.

En el SO-101, las acciones de LeRobot suelen representar objetivos de posición para los seis motores. No son directamente pares ni torsiones.

## Training e inference

### Training

```text
Dataset
  ↓
Modelo
  ↓
Ajustes internos
```

### Inference

```text
Observación actual
  ↓
Modelo entrenado
  ↓
Acción
  ↓
Nueva observación
  ↓
Siguiente acción
```

Una pérdida baja durante el entrenamiento no significa que la policy complete la tarea. Solo la evaluación en el robot demuestra su comportamiento real.

---

# 10. Ejemplo completo de Imitation Learning

## Objetivo

> "Coge el cubo rojo y colócalo dentro de una caja."

## Paso 1: preparar el escenario

Necesitamos:

```text
Mesa
Cubo rojo
Caja
SO-101
Cámara
```

La cámara debe mostrar claramente:

- el cubo;
- el gripper;
- la caja.

## Paso 2: demostrar

El humano realiza la tarea muchas veces.

Podemos empezar con:

- 10 demostraciones para comprobar el proceso;
- 30 para obtener un primer dataset;
- 50 o más si queremos cubrir cinco posiciones.

La cantidad depende de:

- dificultad;
- variedad;
- calidad;
- modelo;
- cámaras;
- estabilidad del robot.

No existe un número mágico.

## Paso 3: crear el dataset

Cada intento queda registrado.

```text
Imágenes
+
Estado
+
Acciones
+
Tarea
+
Tiempo
+
Episodio
```

## Paso 4: entrenar

La policy aprende patrones como:

```text
Cuando veo el cubo en esta zona
y el gripper está abierto,
debo moverme así,
cerrar cuando llegue aquí
y continuar hacia la caja.
```

## Paso 5: inferencia

Retiramos al humano.

```text
Cámara + estado
       ↓
Policy
       ↓
Primer movimiento
       ↓
Nueva imagen
       ↓
Siguiente movimiento
```

## Paso 6: evaluar

Hacemos diez intentos:

- 7 correctos;
- 3 fallidos.

```text
Success rate = 7 / 10 = 70 %
```

Conviene utilizar condiciones reproducibles:

- mismas posiciones;
- mismo objeto;
- misma cámara;
- mismo criterio de éxito.

## Paso 7: analizar errores

Preguntas importantes:

- ¿El cubo estaba demasiado lejos?
- ¿La iluminación cambió?
- ¿El grasp fue incorrecto?
- ¿La posición nunca apareció en entrenamiento?
- ¿La cámara se movió?
- ¿El robot soltó el objeto demasiado pronto?
- ¿La policy entendió la tarea o memorizó una trayectoria?

## Paso 8: mejorar los datos

No recogemos demostraciones aleatorias. Recogemos nuevas demostraciones donde falla:

```text
Error por posición lejana
  ↓
Más demos con esa zona

Error de iluminación
  ↓
Más demos con distintas luces

Grasp fallido
  ↓
Más ejemplos de grasp centrados

Caída del objeto
  ↓
Demostraciones de verify y retry
```

El ciclo es:

```text
Recoger
  ↓
Entrenar
  ↓
Evaluar
  ↓
Analizar
  ↓
Corregir datos
  ↓
Evaluar de nuevo
```

---

# 11. Visión robótica

## Tener una cámara no significa entender

Una cámara produce píxeles.

```text
Imagen → números
```

Comprender una escena significa convertir esos números en información útil:

```text
Imagen
  ↓
Cubo rojo en la posición X
Caja en la posición Y
Gripper abierto
```

## Nivel 1: la imagen entra directamente en la policy

```text
Imagen + estado
       ↓
Policy
       ↓
Acción
```

El modelo aprende a relacionar píxeles y movimientos.

Es el enfoque más sencillo para empezar con imitation learning.

## Nivel 2: un detector identifica objetos

Un detector o clasificador puede producir:

```text
Detectado:
- cubo rojo: 92 % de confianza
- cubo azul: 87 %
- caja: 95 %
- posición: x=..., y=...
```

Después, otra lógica decide:

```text
El objetivo es rojo
  ↓
Usar la posición del cubo rojo
```

Este enfoque facilita:

- verificar detecciones;
- entender fallos;
- cambiar la policy;
- añadir reglas de seguridad.

## Nivel 3: un modelo multimodal interpreta la escena

Un VLM podría describir:

> "Hay tres objetos. El rojo está detrás del azul y la caja está a la derecha."

Esto permite comprender:

- relaciones espaciales;
- lenguaje;
- instrucciones;
- ambigüedades.

No es necesario para nuestro primer prototipo.

## Relación con nuestro proyecto

Podemos empezar así:

```text
Cámara
  ↓
Policy visual
  ↓
Acción
```

Y evolucionar:

```text
Cámara
  ↓
Detector
  ↓
Regla de selección
  ↓
Policy de grasp
```

El segundo nivel puede ser más fácil de depurar y evaluar.

---

# 12. Una cámara o varias cámaras

## Cámara frontal

Ve bien la mesa, los objetos y el destino.

Ventajas:

- contexto general;
- útil para colores;
- permite comprobar el resultado.

Problemas:

- el brazo puede tapar el objeto;
- difícil inferir profundidad;
- útil para manipulación.

## Cámara superior

Ve posiciones y relaciones espaciales.

Ventajas:

- buena para grasp;
- permite medir mejor las posiciones;
- ayuda con oclusiones.

Problemas:

- perspectiva distorsionada;
- no ve bien la altura;
- puede ver peor el gripper.

## Cámara de muñeca

Está cerca del efector final.

Ventajas:

- permite ver el contacto;
- ayuda a alinear;
- útil cuando el brazo tapa otras vistas.

Problemas:

- se mueve con el robot;
- su imagen cambia continuamente;
- puede perder la mesa completa.

## Combinación

Una configuración frecuente es:

```text
Cámara frontal
  +
Cámara superior
```

o:

```text
Cámara superior
  +
Cámara de muñeca
```

## Ejemplos reales

- [SmolVLA SO-100](https://huggingface.co/datasets/lerobot/svla_so100_pickplace): top y wrist.
- [Teleop Cubes](https://huggingface.co/datasets/qb1t/so101_teleop_cubes): frontal y wrist.
- [SO-101 PickPlace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace): up y side.
- [Sponge with Distractors](https://huggingface.co/datasets/aswinkumar99/LeRobot-SO101-task3-single-sponge-with-distractors-random-locations): overhead y wrist.
- [ArmnetBench](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101): front, top y wrist.
- El [proyecto ACT de Sherry Chen](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) cambió de front/side a front/top al analizar oclusiones durante el grasp.

## Nuestra recomendación inicial

Empezar con:

- una cámara superior o frontal bien fijada;
- iluminación estable;
- una sola tarea.

Después añadir una segunda cámara si existe un problema claro:

> "El robot no sabe cuándo sujetar el objeto".

En ese caso puede ayudar una wrist camera.

Más cámaras no significan automáticamente mejor rendimiento. Aumentan:

- datos;
- sincronización;
- consumo;
- complejidad;
- posibilidad de desajustes.

---

# 13. Simulación

## Qué es una simulación

Es un entorno controlado que imita alguna parte del mundo físico.

Podemos probar:

- movimiento;
- observaciones;
- acciones;
- recompensas;
- colisiones;
- policies;
- generación de datos.

## Isaac Sim, Isaac Lab y LeIsaac

- **Isaac Sim** proporciona el entorno de simulación.
- **Isaac Lab** ofrece herramientas para entornos y robot learning.
- [LeIsaac](https://github.com/LightwheelAI/leisaac) integra SO-101, teleoperación, grabación y conversión a formato LeRobot.

## Qué podemos probar

- que el loop de observación y acción funcione;
- distintos escenarios;
- distintas posiciones;
- muchas repeticiones;
- aleatorización;
- una policy antes de usarla en hardware.

## Qué no representa perfectamente

La simulación no reproduce exactamente:

- fricción;
- deformación de la pinza;
- cables;
- backlash;
- fallos de motores;
- diferencias entre SO-101;
- iluminación real;
- reflejos;
- oclusiones inesperadas;
- latencia de cámaras;
- errores de calibración;
- contactos suaves.

## Relación con `demos/01_simulation_basics`

Nuestro demo actual representa el primer peldaño: **entender el ciclo básico de un entorno**.

La intención del demo está documentada en `demos/01_simulation_basics/README.md:3-7`.

El entorno cargado es:

```text
LeIsaac-SO101-LiftCube-v0
```

Es una tarea de levantar un cubo. No es todavía una escena general de pick-and-place con una zona objetivo configurable, como reconoce `demos/01_simulation_basics/README.md:100-106`.

Los tres scripts actuales:

1. comprueban imports;
2. crean e inspeccionan el entorno;
3. envían acciones fijas pequeñas.

La acción constante de `0.1` aparece en `demos/01_simulation_basics/03_basic_actions.py:55-63`.

Esto nos ayuda a entender:

```text
Entorno
   ↓
Observación
   ↓
Acción
   ↓
Nuevo estado
```

Todavía no demuestra:

- que el cubo sea agarrado;
- que la policy sea correcta;
- que exista una zona objetivo;
- que el robot real pueda replicarlo.

## Qué aporta al proyecto

La simulación sirve para:

- aprender la interfaz;
- iterar rápido;
- medir una policy;
- construir escenarios controlados;
- separar el algoritmo del hardware.

Su papel inicial es educativo y experimental, no un atajo mágico.

---

# 14. Sim-to-real

## Simulación

En simulación podemos controlar muchas variables:

- posición exacta;
- iluminación;
- fricción;
- peso;
- cámaras;
- aleatorización.

## Realidad

En el robot real aparece:

- ruido de sensores;
- vibración;
- pequeñas diferencias de calibración;
- delays;
- cables;
- objetos deformables;
- iluminación cambiante;
- fricción diferente;
- errores de grasp;
- motores y cables con límites.

## Por qué falla la transferencia

Una policy puede aprender:

> "Cuando la imagen se parece a esta, muevo el brazo así".

Pero si en el robot real la cámara está dos centímetros más alta, esa correspondencia cambia.

Otro ejemplo:

```text
Simulación:
El cubo queda exactamente entre los dedos

Realidad:
El cubo se desplaza y un dedo lo empuja
```

## Cómo reducir el salto

- usar datos reales;
- aleatorizar la simulación;
- medir parámetros;
- añadir diferentes posiciones;
- probar varias luces;
- fijar y documentar cámaras;
- evaluar fase por fase;
- hacer fine-tuning con datos reales.

La conclusión importante es:

> Funcionar perfectamente en simulación no significa funcionar perfectamente en el robot real.

La simulación es un laboratorio. El robot real es la validación final.

---

# 15. Proyectos reales de la comunidad

## 1. Tutorial oficial ACT + SO-101

**Nivel: sencilla.**

[Documentación de Imitation Learning con SO-101](https://huggingface.co/docs/lerobot/il_robots)

- **Qué hace:** recoge una tarea, registra imágenes, estado y acciones, entrena ACT y ejecuta la policy.
- **Interés:** es la ruta de aprendizaje más directa.
- **Qué reutilizar:** una tarea pequeña, una cámara estable y una definición clara de éxito.
- **Límite:** primero hay que conseguir fiabilidad en el grasp.

## 2. ACT on SO-101: lessons de una comunidad

**Nivel: media.**

[Artículo y código](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) · [GitHub](https://github.com/sherrychen1120/so101_bench)

- **Qué hace:** pick-and-place de bloques mediante ACT.
- **Qué muestra:** una cámara movida, una calibración incorrecta y datos mal distribuidos pueden provocar que el robot "picotee" la mesa.
- **Resultado declarado por la autora:** 90 % dentro de distribución y 75 % fuera de distribución después de mejorar los datos.
- **Qué reutilizar:** diseñar condiciones de evaluación antes de entrenar y estudiar los datos, no solo la loss.

Las cifras son autorreportadas y no constituyen un benchmark independiente.

## 3. SmolVLA

**Nivel: avanzada.**

[Blog oficial](https://huggingface.co/blog/smolvla) · [Documentación](https://huggingface.co/docs/lerobot/smolvla)

- **Qué hace:** combina múltiples cámaras, estado e instrucción de lenguaje para producir acciones.
- **Interés:** muestra una ruta hacia policies generalistas y condicionadas por lenguaje.
- **Qué reutilizar:** estandarizar vistas de cámara y hacer fine-tuning con datos propios.
- **Límite:** requiere un dataset coherente y más capacidad computacional.

## 4. Naive Bench

**Nivel: media/avanzada.**

[Dataset y benchmark](https://huggingface.co/datasets/binhpham/naive-bench)

- **Qué hace:** sorting de barras según dos elementos de lenguaje: color del objeto y color del destino.
- **Interés:** evalúa si la policy entiende la instrucción o solo memoriza.
- **Qué reutilizar:** conjunto reservado de combinaciones, cámaras separadas y puntuación por etapas.
- **Límite:** el conjunto completo y la evaluación son más exigentes que un pick-and-place inicial.

## 5. Project IRA

**Nivel: media/avanzada.**

[GitHub](https://github.com/Project-IRA/interactive-robotic-arm)

- **Qué hace:** pipeline completo sobre SO-101 con interfaz, teleoperación, grabación, playback, entrenamiento e inferencia.
- **Modos publicados:** Desk Cleanup, Lego Sorter, Dice Roller y Play Fetch.
- **Qué reutilizar:** separación entre grabación, entrenamiento, inferencia y aplicación.
- **Límite:** es un proyecto grande; no conviene replicarlo entero durante un hackathon.

## 6. NVIDIA GR00T sobre SO-101

**Nivel: avanzada.**

[Tutorial oficial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)

- **Qué hace:** post-training de GR00T para table cleanup con imágenes e instrucciones.
- **Interés:** muestra cómo adaptar un foundation model a un embodiment no incluido originalmente.
- **Qué reutilizar:** separar cliente robótico y servidor de inferencia.
- **Límite:** necesita GPU, conversión de dataset y más ingeniería.

## 7. HIL-SERL con SO-101

**Nivel: avanzada.**

[GitHub](https://github.com/ggand0/hil-serl-so101) · [Paper](https://arxiv.org/abs/2410.21845)

- **Qué hace:** reinforcement learning online con intervenciones humanas y clasificación visual de recompensa.
- **Interés:** combina datos automáticos y correcciones humanas.
- **Resultado declarado:** alrededor de 70 % de grasp autónomo tras unas tres horas y aproximadamente 600 episodios.
- **Límite:** el proyecto se centra en grasp, no en un pick-and-place completo.

## 8. LeIsaac

**Nivel: avanzada para simulación, media para uso.**

[GitHub](https://github.com/LightwheelAI/leisaac)

- **Qué hace:** integra SO-101 en IsaacLab, leader/follower, teleoperación, data collection y conversión de datos.
- **Tareas publicadas:** levantar cubos, recoger objetos, limpiar una mesa y otras manipulaciones.
- **Qué reutilizar:** mantener una interfaz común entre simulación y robot real.
- **Límite:** la transferencia y el comportamiento de los materiales siguen necesitando validación real.

## 9. Sim-to-Real SO-101 Workshop

**Nivel: avanzada.**

[GitHub](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)

- **Qué hace:** flujo Isaac Sim/Isaac Lab para recoger, entrenar y evaluar la manipulación de un vial y colocarlo en un rack.
- **Interés:** compara escenarios con y sin domain randomization.
- **Qué reutilizar:** separar entrenamiento, evaluación y despliegue real.
- **Límite:** requiere una infraestructura de simulación considerablemente más compleja.

## 10. ArmnetBench

**Nivel: avanzada.**

[Paper](https://arxiv.org/abs/2607.24481) · [Dataset](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101)

- **Qué hace:** evalúa siete policies en SO-101 sobre varias tareas.
- **Interés:** etiqueta éxitos, fallos y comportamientos subóptimos.
- **Qué reutilizar:** evaluación por etapas, etiquetas de calidad y comparación de múltiples modelos.
- **Límite:** es un benchmark, no una demo sencilla de un solo brazo.

## 11. Strands Robots + LeRobot

**Nivel: avanzada.**

[Artículo](https://huggingface.co/blog/amazon/strands-lerobot-hub-to-hardware) · [GitHub](https://github.com/strands-labs/robots)

- **Qué hace:** conecta un agente con grabación, simulación, policies y hardware.
- **Interés:** el agente orquesta herramientas mientras LeRobot conserva el control físico.
- **Qué reutilizar:** separar un agente de alto nivel de las skills de bajo nivel.
- **Límite:** el ejemplo por defecto usa una policy simulada (mock) y no demuestra por sí mismo un grasp real.

---

# 16. Casos de uso que podríamos construir

El orden no es un ranking.

## 1. Pick-and-place inteligente

**Qué hace:** coge un objeto y lo coloca en un destino.

**Qué ve:** cubo, destino y gripper.

**Qué aprende:** approach, grasp, lift y place.

**Parte programada:** posiciones fijas, zonas de destino y comprobaciones básicas.

**Parte ML:** policy de manipulación.

**Qué demostrar:** cinco o diez intentos con tasa de éxito medida.

**Dificultad:** sencilla.

## 2. Sorting

**Qué hace:** separa objetos según color, forma o estado.

**Qué ve:** varios objetos y varias categorías.

**Qué aprende:** seleccionar el objeto correcto y moverlo al destino correspondiente.

**Parte programada:** reglas de destino y detección de categorías.

**Parte ML:** visión y policy de pick-and-place.

**Qué demostrar:** dos colores o dos formas, con una matriz clara de aciertos.

**Dificultad:** sencilla o media.

## 3. Clean the table

**Qué hace:** retira varios objetos de una mesa y los coloca en zonas.

**Qué ve:** desorden, varios objetos y una posible caja.

**Qué aprende:** reconocer qué debe moverse, aproximación y colocación.

**Parte programada:** orden de ejecución, zonas y límites de intentos.

**Parte ML:** detección y policy de grasp.

**Qué demostrar:** tres objetos con una única caja, sin cambiar la tarea durante la demo.

**Dificultad:** media.

## 4. Object retrieval

**Qué hace:** responde a "pásame el objeto rojo" dejándolo en una bandeja.

**Qué ve:** varios objetos y la bandeja.

**Qué aprende:** vincular lenguaje, color y manipulación.

**Parte programada:** menú cerrado de instrucciones y verificación.

**Parte ML:** visión y policy; posteriormente, condicionamiento por lenguaje.

**Qué demostrar:** una instrucción distinta de la utilizada en la teleoperación.

**Dificultad:** media.

La entrega directa en la mano debe evitarse salvo que se hayan definido medidas de seguridad. Para el hackathon es más seguro dejarlo en una bandeja.

## 5. Assembly assistant

**Qué hace:** entrega piezas en un orden o las coloca en varias zonas.

**Qué ve:** pieza actual y contenedor correspondiente.

**Qué aprende:** approach y colocación repetibles.

**Parte programada:** secuencia y estado del pedido.

**Parte ML:** policy de manipulación.

**Qué demostrar:** tres piezas en el orden correcto.

**Dificultad:** media.

## 6. Quality-control assistant

**Qué hace:** identifica una pieza con una marca o estado visual y la mueve a una zona de aceptada o rechazada.

**Qué ve:** color, marca, forma o defecto claramente visible.

**Qué aprende:** grasping y clasificación visual.

**Parte programada:** reglas de clasificación y zonas.

**Parte ML:** detector y policy.

**Qué demostrar:** una regla sencilla y visible, como una pieza azul con una marca roja.

**Dificultad:** media.

## 7. Juego de instrucciones

**Qué hace:** ordena piezas siguiendo una instrucción, por ejemplo "rojo, azul, rojo".

**Qué ve:** piezas y zona de entrega.

**Qué aprende:** seguir una secuencia y mantener el estado.

**Parte programada:** intérprete sencillo de instrucciones y estado de secuencia.

**Parte ML:** visión y manipulación.

**Qué demostrar:** tres instrucciones diferentes sin cambiar el hardware.

**Dificultad:** media.

## 8. Aplicación de accesibilidad o asistencia

**Qué hace:** trae y coloca un objeto solicitado en una zona accesible.

**Qué ve:** objeto, destino y persona o bandeja.

**Qué aprende:** manipulación y, más adelante, comprensión del lenguaje.

**Parte programada:** interfaz de botones, lista cerrada de objetos y zonas.

**Parte ML:** detección de color y policy.

**Qué demostrar:** una persona puede pedir uno de varios objetos mediante una interfaz sencilla.

**Dificultad:** media.

## 9. Pick-and-place con distractores

**Qué hace:** escoge un objeto específico entre otros parecidos o visualmente distractores.

**Qué ve:** varios objetos con colores, formas o posiciones diferentes.

**Qué aprende:** seleccionar el objetivo y no confundirlo con otro.

**Parte programada:** categorías y reglas de seguridad.

**Parte ML:** detector y policy.

**Qué demostrar:** dos o tres distractores y una tasa de éxito medida.

**Dificultad:** media o avanzada.

## 10. Tarea con verificación y recuperación

**Qué hace:** intenta completar una tarea, comprueba el resultado y reintenta si falla.

**Qué ve:** estado de la escena antes y después de cada acción.

**Qué aprende:** la habilidad principal; la recuperación puede ser inicialmente reglas.

**Parte programada:** verificar, retroceder, esperar y reintentar.

**Parte ML:** policy de grasp/place.

**Qué demostrar:** al menos un fallo provocado y una recuperación visible.

**Dificultad:** media o avanzada.

---

# 17. Nuestro concepto: SO-101 Autonomous Agent

La idea del proyecto es:

```text
VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE
```

## VER

### Qué información obtenemos

- imágenes de la cámara;
- posición de las articulaciones;
- estado del gripper;
- hora;
- tarea;
- posiblemente presencia de objetos.

Ejemplo:

> La cámara muestra tres objetos: rojo, azul y verde. El brazo está quieto.

## ENTENDER

### Qué significa la escena

El robot debe convertir la imagen en información útil:

```text
Objeto rojo localizado
Objeto azul localizado
Caja visible
Gripper vacío
```

Puede resolverse mediante:

- detector;
- clasificador de color;
- modelo de visión;
- policy visual implícita.

Ejemplo:

> Hay un cubo rojo en la zona izquierda y una caja en la zona derecha.

## PLANIFICAR

### Qué pasos hay que realizar

```text
1. Localizar el cubo.
2. Acercar el gripper.
3. Abrir.
4. Bajar.
5. Cerrar.
6. Levantar.
7. Mover hacia la caja.
8. Soltar.
```

La planificación puede ser una lista de reglas simple o un plan generado por un sistema más complejo.

## ACTUAR

### El robot ejecuta movimientos

Puede usar:

- una policy;
- una secuencia programada;
- un controlador clásico;
- una combinación.

## VERIFICAR

### Comprobamos qué ocurrió

Preguntas:

- ¿El cubo está en la pinza?
- ¿El gripper está vacío?
- ¿El cubo está en la caja?
- ¿La caja sigue en el lugar esperado?
- ¿La tarea terminó?

Ejemplo:

> El cubo ya no está sobre la mesa y aparece dentro de la caja.

La verificación necesita una condición clara. "El robot dejó de moverse" no significa que haya terminado.

## RECUPERARSE

### Qué hacemos si falla

```text
Grasp fallido
   ↓
Abrir gripper
   ↓
Retroceder
   ↓
Observar de nuevo
   ↓
Localizar el objeto
   ↓
Nuevo intento
```

La recuperación no tiene que ser infinitamente inteligente. Puede ser:

- un número máximo de reintentos;
- una retirada segura;
- una nueva observación;
- una solicitud de ayuda humana.

## Arquitectura progresiva recomendada

```text
Nivel 1:
Teleoperación + grabación

Nivel 2:
Policy para una tarea fija

Nivel 3:
Policy que responde a color o posición

Nivel 4:
Planificador con varios pasos

Nivel 5:
Verificación de resultado

Nivel 6:
Un reintento sencillo
```

No necesitamos un LLM, un VLM y un VLA para implementar este proyecto.

---

# 18. No convertir todo en IA

Un buen robot puede combinar varias técnicas.

| Problema | Técnica adecuada |
|---|---|
| Seguir un orden fijo | Reglas |
| Mantener una secuencia | Máquina de estados |
| Controlar una trayectoria | Control clásico |
| Detectar colores | Visión clásica o detector pequeño |
| Imitar una manipulación | Imitation learning |
| Seguir una instrucción abierta | VLA o policy condicionada por lenguaje |
| Comprobar un resultado | Regla y visión |
| Reintentar | Lógica determinista |

## Ejemplo

Una máquina de estados puede decidir:

```text
buscar
  ↓
agarrar
  ↓
comprobar
  ↓
colocar
  ↓
comprobar
```

Una policy aprendida puede ejecutar:

```text
grasp(red_cube)
```

Un detector puede decidir:

```text
El objeto seleccionado es el cubo rojo
```

## Regla útil

> IA donde aporta valor, lógica determinista donde sea suficiente.

No usamos machine learning porque "se podría", sino porque existe una incertidumbre o una habilidad que resulta difícil de codificar.

## Qué aporta realmente el ML

El ML suele ser útil para:

- mapear imagen y estado a acciones;
- reconocer variaciones visuales;
- imitar movimientos;
- aprender una habilidad de grasp;
- seleccionar un objeto según lenguaje.

## Qué no necesita ML

No siempre necesita ML:

- detectar que el episodio terminó;
- abrir el gripper después de colocarlo;
- incrementar un contador;
- repetir una tarea tres veces;
- regresar a casa;
- pedir ayuda si se superan los reintentos.

Esta mezcla hace que el sistema sea más comprensible y depurable.

---

# 19. Skills robóticas

En lugar de crear una "inteligencia gigante", podemos construir habilidades pequeñas.

Ejemplos:

```text
home
find_object
approach_object
grasp
lift
place
verify
retreat
retry
```

## Ejemplo conceptual

Tarea:

> "Recoge el cubo rojo".

Flujo:

```text
find_object(red)
       ↓
approach_object(red)
       ↓
grasp(red)
       ↓
lift
       ↓
place(box)
       ↓
verify()
```

## Qué aporta cada skill

- `home`: devuelve un estado conocido y seguro;
- `find_object`: devuelve una posición o "no encontrado";
- `approach_object`: acerca el efector final;
- `grasp`: intenta sujetar;
- `lift`: eleva el objeto;
- `place`: transporta y suelta;
- `verify`: comprueba el resultado;
- `retry`: decide si puede volver a intentarlo.

Una skill puede tener:

- una acción;
- una condición de éxito;
- una condición de fallo;
- un tiempo máximo;
- un estado de seguridad.

## Relación con nuestro proyecto

El agente no necesita controlar cada articulación directamente. Puede coordinar habilidades:

```text
Agente
  ├── seleccionar objetos
  ├── seleccionar skills
  ├── comprobar resultados
  └── solicitar recuperación

Skills
  ├── grasp
  ├── lift
  ├── place
  └── verify
```

Así podemos sustituir una policy por otra sin rehacer todo el sistema.

---

# 20. Del robot controlado al robot autónomo

## Nivel 0: movimiento manual

El humano mueve directamente el robot.

```text
Teleoperación
```

## Nivel 1: secuencia programada

El robot sigue movimientos definidos.

```text
home → approach → close → lift → place
```

## Nivel 2: teleoperación

Un humano controla la tarea completa.

## Nivel 3: imitation learning

El robot imita demostraciones.

```text
Imágenes + estado
       ↓
Policy
       ↓
Movimiento
```

## Nivel 4: visión

Las acciones dependen de lo que se observa.

El cubo puede estar en varias posiciones.

## Nivel 5: variabilidad

El robot puede trabajar con:

- distintas posiciones;
- distintos ángulos;
- pequeñas variaciones de iluminación;
- varios objetos.

## Nivel 6: instrucciones

El robot recibe una tarea:

> "Coloca el cubo rojo en la caja azul".

## Nivel 7: verificación

El robot comprueba si la tarea se completó.

## Nivel 8: recuperación

El robot intenta corregir un fallo.

```text
Intentar
  ↓
Verificar
  ↓
¿Correcto?
  ├── Sí → terminar
  └── No → reintentar o pedir ayuda
```

No hace falta llegar al nivel 8 para tener un buen proyecto. Un sistema en nivel 4 o 5 ya puede ser una demo con criterio.

---

# 21. Qué deberíamos preparar antes del hackathon

Sin convertir esta lista en una guía de instalación.

## Conocimientos

- entender qué es un SO-101;
- entender leader y follower;
- conocer observaciones, estados y acciones;
- saber qué es un episode;
- distinguir training de inference;
- entender behavior cloning;
- conocer la diferencia entre ACT, language-conditioned policy y VLA;
- entender qué es sim-to-real.

## Experimentos

- teleoperar un SO-101;
- comprobar una calibración;
- grabar unos pocos episodios;
- visualizar el dataset;
- probar una policy sencilla;
- ejecutar una tarea en simulación;
- medir una tasa de éxito;
- repetir la misma condición diez veces;
- observar qué gripper falla y por qué.

## Investigación

- estudiar los datasets de esta guía;
- comparar una cámara frontal, superior y de muñeca;
- leer el tutorial oficial de ACT;
- revisar SmolVLA sin intentar implementarlo todo;
- estudiar recuperación y verificación;
- revisar proyectos de la comunidad.

## Arquitectura

Preparar una estructura modular:

```text
Robot adapter
Observation
Skills
Policy
Task logic
Evaluation
Visualization
```

## Demostraciones

Tener demos pequeñas y explicables:

```text
01 — Simulation basics
02 — Simple robot control
03 — Teleoperation
04 — Record demonstrations
05 — Understand a dataset
06 — Train a simple policy
07 — Evaluate a policy
08 — Vision experiments
09 — Autonomous task
10 — Recovery
```

No hace falta que todas existan antes del hackathon. Sí conviene saber qué demostraría cada una.

---

# 22. Qué tiene sentido hacer durante el hackathon

La filosofía debe ser progresiva.

## Primero: una tarea mínima funcionando

Ejemplo:

> Coger un cubo de una posición conocida y colocarlo en una caja.

No:

> Construir un agente generalista con lenguaje, memoria, varios modelos y recuperación ilimitada.

## Después: hacerla repetible

Que no funcione solo una vez.

Hay que repetir:

- misma tarea;
- mismas condiciones;
- diferentes intentos;
- preparación manual controlada.

## Después: medirla

Registrar:

- éxitos;
- fallos;
- fases;
- tiempo;
- número de reintentos;
- causas de error.

## Después: añadir inteligencia

Por ejemplo:

- visión para distinguir colores;
- dos posiciones;
- dos objetos;
- lenguaje limitado;
- un segundo objeto en la escena.

## Finalmente: mejorar la demo

Añadir:

- interfaz visual sencilla;
- vídeo de la cámara;
- explicación de la arquitectura;
- comparación antes/después;
- contador de resultados;
- fallback si el robot falla.

## Regla de trabajo

```text
Tarea mínima
  ↓
Repetible
  ↓
Medible
  ↓
Más inteligente
  ↓
Más visual
```

No se avanza a la siguiente capa hasta que la anterior sea demostrable.

---

# 23. Qué no deberíamos hacer

No deberíamos:

- intentar construir un robot generalista;
- empezar por un agente extremadamente complejo;
- utilizar un LLM para todo;
- combinar cinco modelos diferentes;
- entrenar un modelo gigante sin datos propios;
- intentar resolver diez tareas;
- cambiar de objetivo cada pocas horas;
- perseguir una tecnología sin una demo clara;
- medir éxito solo por la loss;
- usar un simulador como prueba final;
- ocultar que el robot necesita ayuda manual;
- presentar una tarea espectacular sin repeticiones;
- hacer que el robot opere sin límites de seguridad.

El objetivo no es demostrar que sabemos mencionar muchos nombres. Es demostrar una capacidad concreta.

---

# 24. Cómo evaluar una idea de proyecto

No hace falta asignar una puntuación automática. Estas preguntas ayudan a razonar.

## Claridad

- ¿Se puede explicar en una frase?
- ¿Se entiende qué hace el robot?
- ¿Se entiende qué observa?
- ¿La demo tiene un principio y un final claros?

## Valor

- ¿Utiliza realmente el robot?
- ¿La IA aporta algo útil?
- ¿La parte de machine learning podría sustituirse fácilmente por una regla?
- ¿El proyecto demuestra una habilidad interesante?

## Viabilidad

- ¿Tiene un MVP muy pequeño?
- ¿Podemos recoger suficientes datos?
- ¿La tarea cabe en el tiempo disponible?
- ¿El hardware permite realizarla?
- ¿La escena está controlada?
- ¿La policy se puede evaluar en un tiempo razonable?

## Medición

- ¿Podemos saber si tuvo éxito?
- ¿Podemos saber si agarró el objeto?
- ¿Podemos diferenciar "no lo encontró" de "no lo agarró"?
- ¿Podemos repetir la prueba?
- ¿Tenemos una condición de fallo clara?

## Progresividad

- ¿Podemos añadir mejoras sin reconstruirlo todo?
- ¿La visión es una mejora opcional?
- ¿La comprensión de lenguaje se puede añadir después?
- ¿La recuperación puede empezar con un solo reintento?
- ¿La demo sigue siendo clara si una parte falla?

## Repetición

- ¿Podemos demostrarlo varias veces?
- ¿La tasa de éxito es comprensible?
- ¿La tarea puede fallar de forma realista?
- ¿La persona que evalúa entiende qué está mirando?

Una buena tarea se puede resumir así:

> "El SO-101 recibe una instrucción sencilla, elige un objeto visible, ejecuta una manipulación aprendida, comprueba el resultado y reintenta una vez si el grasp falla."

---

# 25. Análisis de nuestro repositorio actual

## README principal

El README ya define la visión del proyecto:

- SO-101 y LeRobot;
- Hackathon Robotekin 2026;
- el loop `VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE`;
- autonomous agent;
- multi-robot como extensión.

El bucle aparece en `README.md:7-9`.

El primer objetivo declarado es conseguir que un SO-101 simulado coja un cubo y lo coloque en una zona determinada, en `README.md:79-85`.

La estrategia general aparece en `README.md:127-141`:

```text
SO-101 simulado
  ↓
Teleoperación
  ↓
Demostraciones
  ↓
Policies
  ↓
Ejecución autónoma
  ↓
SO-101 real
```

El multi-robot aparece como idea futura en `README.md:87-125`, pero no está implementado.

## Demo `01_simulation_basics`

El objetivo del demo es aprender la interfaz básica de una simulación de SO-101, no resolver todavía el proyecto completo.

La documentación lo define en `demos/01_simulation_basics/README.md:3-7`.

### Qué hace realmente

1. Comprueba si están disponibles las dependencias.
2. Carga un entorno externo de LeIsaac mediante EnvHub.
3. Reinicia el entorno.
4. Muestra observaciones y espacios de acción.
5. Aplica acciones fijas pequeñas.
6. Imprime recompensas, terminación e información del entorno.

El script de acciones utiliza vectores de ceros y valores constantes de `0.1` en `demos/01_simulation_basics/03_basic_actions.py:55-63`.

### Qué no hace

- No implementa una policy.
- No controla una pinza de forma específica de la tarea.
- No encuentra el cubo mediante visión.
- No coloca el cubo en una zona.
- No graba un dataset.
- No entrena un modelo.
- No verifica una tasa de éxito.
- No implementa el agente autónomo.

La propia documentación reconoce que el task actual es `LiftCube` y que todavía no existe un pick-and-place genérico, en `demos/01_simulation_basics/README.md:100-106`.

## Estado de Git

En el momento de la revisión:

- el README principal está versionado;
- los tres scripts y el README del demo están sin trackear;
- los logos también están sin trackear;
- el remoto configurado es `https://github.com/iker-azur/hackathon-robotekin.git`;
- ese remoto no era accesible públicamente sin autenticación en la fecha de consulta.

Por tanto, el repositorio local es un punto de partida conceptual y experimental, no todavía una implementación pública completa del agente.

## Dónde encaja el demo

Está en la fase inicial:

```text
Comprender el ciclo:
observación → acción → resultado
```

Antes de entrenar una policy necesitamos entender:

- qué observaciones entrega el entorno;
- qué significa cada acción;
- cómo se termina un episodio;
- qué se considera éxito.

## Demos conceptuales progresivos

### 01 — Simulation Basics

Entender el entorno, las observaciones y las acciones.

### 02 — Simple Robot Control

Realizar una secuencia pequeña, como `home` o levantar un cubo.

### 03 — Teleoperation

Mover el follower con el leader y comprobar la correspondencia.

### 04 — Record Demonstrations

Grabar una tarea completa y revisar qué se ha registrado.

### 05 — Understand a Dataset

Visualizar imágenes, estados, acciones, tiempo y episodes.

### 06 — Train a Simple Policy

Entrenar ACT u otra policy sencilla con un conjunto pequeño.

### 07 — Evaluate a Policy

Ejecutar diez intentos y medir éxitos/fallos.

### 08 — Vision Experiments

Probar detección de color, posiciones o comprensión de imágenes.

### 09 — Autonomous Task

Ejecutar una tarea sin humano después de cada movimiento.

### 10 — Recovery

Detectar un fallo, retroceder y reintentar.

La secuencia no exige implementarlas todas antes del hackathon. Sirve para no saltar directamente de la simulación a un agente complejo.

---

# 26. Roadmap conceptual del proyecto

## Fase 1: Comprender

```text
Robot
Teleoperación
Observaciones
Acciones
```

Evidencia: sabemos qué recibe y qué envía el robot.

## Fase 2: Demostrar

```text
Humano realiza una tarea
```

Evidencia: el follower puede repetir una manipulación sencilla.

## Fase 3: Registrar

```text
Demostraciones
  ↓
Dataset
```

Evidencia: existe un episode reproducible con imágenes, estados y acciones.

## Fase 4: Aprender

```text
Dataset
  ↓
Policy
```

Evidencia: el modelo se entrena y produce acciones.

## Fase 5: Ejecutar

```text
Policy
  ↓
Robot
```

Evidencia: el robot actúa sin teleoperación continua.

## Fase 6: Evaluar

```text
10 intentos
  ↓
7 correctos
  ↓
70 %
```

Evidencia: existen resultados reproducibles.

## Fase 7: Generalizar

```text
Misma tarea
Distintas posiciones
Distintos ángulos
```

Evidencia: el robot funciona fuera de la configuración inicial.

## Fase 8: Razonar

```text
Seleccionar objeto
Seleccionar tarea
Seleccionar siguiente skill
```

Evidencia: el sistema puede elegir entre varias opciones controladas.

## Fase 9: Verificar

```text
Observar resultado
Comprobar condición
```

Evidencia: el sistema sabe si terminó correctamente.

## Fase 10: Recuperarse

```text
Fallo
  ↓
Observar
  ↓
Reintentar
```

Evidencia: puede corregir al menos un fallo sencillo.

## Diagrama completo

```text
Comprender
   ↓
Demostrar
   ↓
Registrar
   ↓
Aprender
   ↓
Ejecutar
   ↓
Evaluar
   ↓
Generalizar
   ↓
Razonar
   ↓
Verificar
   ↓
Recuperarse
```

Las fases no son una línea completamente recta. Después de un fallo podemos volver a registrar datos y repetir el ciclo.

---

# 27. Experimentos mentales

## Experimento 1: el cubo solo aparece a la izquierda

Tenemos 30 demostraciones donde el cubo siempre aparece a la izquierda.

¿Qué ocurrirá si durante la prueba lo colocamos a la derecha?

### Respuesta

Es probable que el rendimiento cambie mucho. La policy puede haber aprendido:

> "Cuando veo el objeto en una zona parecida a la izquierda, hago esta trayectoria".

No ha aprendido necesariamente "cubo en cualquier lugar".

Puede funcionar por suerte o por capacidades visuales generales, pero no debemos esperarlo.

### Mejora

Recoger demostraciones en ambas zonas y evaluar explícitamente la posición no vista.

## Experimento 2: movemos la cámara

La policy se entrenó con una cámara fija. Durante la prueba alguien desplaza la cámara unos centímetros.

### Respuesta

La imagen cambia aunque el robot y el objeto sigan igual. La policy puede:

- agarrar el objeto demasiado pronto;
- pasar al lado;
- fallar la aproximación;
- golpear la mesa.

La cámara forma parte del sistema, no es un accesorio neutral.

### Mejora

- fijar la cámara;
- marcar su posición;
- añadir variedad de cámara de forma controlada;
- incluir aumentos de datos solo después de entender el problema.

## Experimento 3: 90 % de éxito en cada grasp

El robot tiene una tasa de éxito del 90 % al agarrar.

Debe realizar tres grasps consecutivos.

### Respuesta

Si los tres son independientes:

```text
0,9 × 0,9 × 0,9 = 0,729
```

Es decir, aproximadamente 72,9 % de éxito en los tres.

Por eso una demo larga puede ser menos fiable que una demo de un solo grasp.

### Mejora

- medir éxito por grasp y por tarea;
- añadir recuperación;
- dividir la tarea en subtareas;
- evitar depender de una sola policy durante muchos movimientos.

## Experimento 4: 50 demostraciones, pero todas con mal grasping

La policy puede aprender a acercarse y mover el brazo, pero no a sujetar correctamente el objeto.

### Respuesta

La calidad de los datos es más importante que su cantidad.

## Experimento 5: 10/10 en posiciones de entrenamiento

El robot acierta todas las posiciones vistas, pero falla en una nueva.

### Respuesta

Puede haber memorizado las trayectorias. El resultado mide reproducción, no generalización.

## Experimento 6: cambiamos la luz

La policy funcionaba con iluminación estable y falla con una luz más débil.

### Respuesta

La imagen ha cambiado. El modelo puede haber asociado colores o posiciones con una iluminación concreta.

### Mejora

Incluir diferentes iluminación en el dataset o controlar y documentar el setup inicial.

---

# 28. Lo que podemos aprender de la comunidad

Los proyectos revisados muestran patrones consistentes.

## 1. Las tareas pequeñas funcionan mejor

La comunidad empieza con:

- un cubo;
- un bloque;
- una caja;
- una pieza;
- un cuenco.

No con un objeto universal.

## 2. El escenario suele estar controlado

La mayoría utiliza:

- una mesa;
- una zona de trabajo pequeña;
- fondos relativamente estables;
- objetos conocidos.

## 3. Las cámaras suelen estar fijadas

El proyecto de [Sherry Chen](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) muestra cómo una cámara movida puede provocar fallos.

## 4. La repetición es fundamental

Las policies se evalúan mediante múltiples intentos, no mediante una única demo.

## 5. Los objetivos son claros

Los mejores ejemplos tienen una condición sencilla:

- objeto dentro de caja;
- pieza dentro de rack;
- color en destino;
- vial colocado.

## 6. Los datasets propios son importantes

Los proyectos comunitarios publican datasets para que otros puedan:

- visualizar;
- entrenar;
- comparar;
- corregir;
- extender.

## 7. Se mejora después de analizar errores

El caso de ACT en SO-101 muestra que no basta con "entrenar más". Hay que revisar:

- grasp;
- ángulos;
- calibración;
- cámaras;
- cobertura de posiciones.

## 8. Las arquitecturas se separan por componentes

Los proyectos más completos separan:

- grabación;
- dataset;
- training;
- inference;
- lógica de tarea;
- hardware.

## 9. La verificación es cada vez más importante

ArmnetBench etiqueta:

- éxito;
- suboptimalidad;
- fallo.

Eso demuestra que no basta con guardar la trayectoria. También hay que saber cómo terminó.

## 10. La recuperación todavía está infradesarrollada

Muchas policies ejecutan una trayectoria y, si falla, repiten el mismo comportamiento. El salto de:

```text
fallo → repetir lo mismo
```

a:

```text
fallo → observar → corregir → reintentar
```

es todavía una oportunidad interesante.

---

# 29. Conclusión: una opción para Robotekin

Una idea concreta y alineada con varios tracks sería:

> Un SO-101 que recibe una instrucción sencilla —por ejemplo, "coge el cubo rojo"—, identifica el objeto mediante una cámara, ejecuta una habilidad de pick-and-place aprendida mediante imitation learning, comprueba visualmente si el objeto llegó a su destino y realiza un reintento controlado si el grasp falla.

Esta idea puede cubrir:

- manipulación de objetos;
- visión robótica;
- aprendizaje por imitación;
- y, si se presenta como tarea de ayuda o accesibilidad, aplicaciones sociales.

No necesitamos empezar con un LLM ni con una arquitectura de agente generalista.

La evolución podría ser:

```text
Teleoperación
  ↓
Pick-and-place aprendido
  ↓
Color o posición variable
  ↓
Varias instrucciones
  ↓
Verificación
  ↓
Recuperación
```

La idea final debe entenderse como una progresión:

> Primero controlar, después demostrar, después aprender, después evaluar y finalmente añadir autonomía.

---

# 30. Glosario sencillo

## SO-101

Brazo robótico abierto y de bajo coste, diseñado para trabajar con LeRobot. Tiene cinco ejes de brazo y un gripper.

## Leader

Brazo o dispositivo que actúa como mando de teleoperación. Sus movimientos se convierten en acciones para el follower.

## Follower

Robot que ejecuta los movimientos enviados por el leader o por una policy.

## Teleoperation

Control remoto de un robot por una persona. Puede grabarse para crear datos.

## Episode

Un intento completo de realizar una tarea. Puede ser exitoso o fallido.

## Dataset

Colección de experiencias robóticas: observaciones, estados, acciones, tareas y metadatos.

## Observation

Información que recibe el robot en un momento concreto: imágenes, estado articular, gripper o sensores.

## Action

Comando que el robot envía para cambiar su estado. En un SO-101 suele ser un objetivo de posición para sus motores.

## Policy

Sistema que decide qué acción realizar a partir de la observación actual.

## Inference

Uso de una policy entrenada para generar acciones sobre el robot o el entorno.

## Training

Proceso de ajustar un modelo con datos para que aprenda una relación entre observaciones y acciones.

## Imitation Learning

Aprendizaje basado en ejemplos de comportamiento, normalmente humanos.

## Behavior Cloning

Forma sencilla de imitation learning que intenta predecir la acción del humano a partir de lo observado.

## ACT

Action Chunking Transformer. Una policy relativamente ligera que predice secuencias de acciones y es una opción inicial habitual en LeRobot.

## VLA

Vision-Language-Action. Modelo que recibe imágenes, estado e instrucción de lenguaje y devuelve acciones del robot.

## SmolVLA

Modelo VLA compacto y abierto publicado por Hugging Face, diseñado para ejecutarse en hardware de bajo coste como SO-100 y SO-101.

## Grasp

Acción de sujetar un objeto con el gripper.

## Pick and place

Tarea de coger un objeto, transportarlo y colocarlo en otro lugar.

## End Effector

Parte final del robot que interactúa con el entorno. En el SO-101 es el gripper.

## Camera observation

Imagen de una cámara utilizada como entrada para que el robot perciba la escena.

## Proprioception

Información interna del propio robot, como las posiciones de sus articulaciones y el estado del gripper.

## Simulation

Entorno artificial para probar robots, acciones, policies o escenarios sin usar el hardware real.

## Sim-to-real

Proceso de trasladar una policy o un comportamiento desde simulación al robot físico.

## Success rate

Porcentaje de intentos en los que la tarea se considera completada correctamente.

## Calibration

Proceso que establece las posiciones y rangos de las articulaciones para que el robot interprete correctamente sus movimientos.

## Fine-tuning

Ajuste de un modelo previamente entrenado con datos específicos de una tarea o robot.

## Rollout

Ejecución completa o parcial de una policy en un entorno o robot.

## State machine

Sistema que cambia de estado según reglas: buscar, agarrar, comprobar, colocar o reintentar.

---

# 31. Referencias

## Robotekin

- [Documentación oficial del hackathon](https://hackathon.robotekin.com/) — documentación oficial
- [Robotekin Hackathon LeRobot 2026 en el sitio de Robotekin](https://robotekin.com/hackathon/) — página oficial
- [Artículo oficial sobre el hackathon](https://robotekin.com/robotekin-hackathon-lerobot-2026/) — artículo
- [Robotekin como asociación](https://robotekin.com/robotekin/) — asociación

## LeRobot

- [Documentación principal de LeRobot](https://huggingface.co/docs/lerobot/index) — documentación oficial
- [Imitation Learning con robots reales](https://huggingface.co/docs/lerobot/il_robots) — documentación oficial
- [LeRobotDataset](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) — documentación oficial
- [Documentación de ACT](https://huggingface.co/docs/lerobot/act) — documentación oficial
- [Documentación de SmolVLA](https://huggingface.co/docs/lerobot/smolvla) — documentación oficial
- [Guía de cámaras](https://huggingface.co/docs/lerobot/cameras) — documentación oficial
- [Repositorio oficial de LeRobot](https://github.com/huggingface/lerobot) — GitHub oficial
- [HF Hub de LeRobot](https://huggingface.co/lerobot) — modelos y datasets

## SO-101 y SO-ARM

- [Documentación oficial del SO-101](https://huggingface.co/docs/lerobot/so101) — documentación oficial
- [Documentación del SO-100](https://huggingface.co/docs/lerobot/so100) — documentación oficial
- [Repositorio de hardware de The Robot Studio](https://github.com/TheRobotStudio/SO-ARM100) — proyecto comunitario

## Datasets interesantes

### Datasets oficiales

- [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) — dataset
- [`lerobot/svla_so100_sorting`](https://huggingface.co/datasets/lerobot/svla_so100_sorting) — dataset
- [`lerobot/svla_so100_stacking`](https://huggingface.co/datasets/lerobot/svla_so100_stacking) — dataset
- [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) — dataset

### Datasets comunitarios

- [`qb1t/so101_teleop_cubes`](https://huggingface.co/datasets/qb1t/so101_teleop_cubes) — dataset comunitario
- [`aswinkumar99/LeRobot-SO101-task3-single-sponge-with-distractors-random-locations`](https://huggingface.co/datasets/aswinkumar99/LeRobot-SO101-task3-single-sponge-with-distractors-random-locations) — dataset comunitario
- [`binhpham/naive-bench`](https://huggingface.co/datasets/binhpham/naive-bench) — dataset comunitario
- [`legalaspro/so101_pickplace_auto_v1`](https://huggingface.co/datasets/legalaspro/so101_pickplace_auto_v1) — dataset comunitario
- [`armnet/armnetbench_v01_lerobot_so101`](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101) — benchmark y dataset

## Proyectos comunitarios

- [ACT on SO-101: lessons](https://huggingface.co/blog/sherryxychen/train-act-on-so-101) — artículo comunitario
- [Código de ACT on SO-101](https://github.com/sherrychen1120/so101_bench) — GitHub
- [Project IRA](https://github.com/Project-IRA/interactive-robotic-arm) — proyecto comunitario universitario
- [HIL-SERL para SO-101](https://github.com/ggand0/hil-serl-so101) — proyecto comunitario
- [Strands Robots + LeRobot](https://huggingface.co/blog/amazon/strands-lerobot-hub-to-hardware) — proyecto de AWS
- [GR00T sobre SO-101](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) — tutorial de NVIDIA
- [LeIsaac](https://github.com/LightwheelAI/leisaac) — proyecto comunitario
- [Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) — material de NVIDIA
- [ArmnetBench](https://huggingface.co/collections/armnet/armnetbench-v01) — benchmark

## Vídeos y demos

- [Demo oficial ACT: recoger un bloque y colocarlo en una caja](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/lerobot_task.mp4) — vídeo
- [Blog y demos de SmolVLA](https://huggingface.co/blog/smolvla) — demos
- [Project IRA: modos y vídeos](https://github.com/Project-IRA/interactive-robotic-arm) — demos
- [Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) — material con demos
- [Playlist oficial de vídeos de LeRobot](https://www.youtube.com/playlist?list=PLo2EIpI_JMQu5zrDHe4NchRyumF2ynaUN) — vídeo
- [Space de ROBO tutorial para visualizar el SO-101 PickPlace](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) — demo interactiva

## Papers para profundizar

- [ACT / ALOHA: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) — paper
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) — paper
- [DAgger](https://arxiv.org/abs/1011.0686) — paper
- [HIL-SERL](https://arxiv.org/abs/2410.21845) — paper
- [SmolVLA](https://arxiv.org/abs/2506.01844) — paper
- [RT-2](https://arxiv.org/abs/2307.15818) — paper
- [OpenVLA](https://arxiv.org/abs/2406.09246) — paper
- [Orbit: A Neural Simulation Framework for Robot Learning](https://arxiv.org/abs/2301.04195) — paper
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — paper
- [Closing the Sim-to-Real Loop](https://arxiv.org/abs/1810.05687) — paper
- [MimicGen](https://proceedings.mlr.press/v229/mandlekar23a.html) — paper

## Repositorios GitHub

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) — repositorio oficial
- [SO-ARM100 / SO-101 hardware](https://github.com/TheRobotStudio/SO-ARM100) — repositorio de hardware
- [LeIsaac](https://github.com/LightwheelAI/leisaac) — repositorio comunitario
- [Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) — repositorio de NVIDIA
- [Project IRA](https://github.com/Project-IRA/interactive-robotic-arm) — repositorio comunitario
- [HIL-SERL SO-101](https://github.com/ggand0/hil-serl-so101) — repositorio comunitario
- [Strands Robots](https://github.com/strands-labs/robots) — repositorio de AWS
- [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — repositorio de NVIDIA
- [SO-101 Bench de Sherry Chen](https://github.com/sherrychen1120/so101_bench) — repositorio comunitario

## Nota sobre nuestro repositorio

El remoto local configurado es `hackathon-robotekin`, pero no era públicamente accesible sin autenticación en la fecha de investigación. El análisis de estado se basa en el directorio local disponible, no en una versión pública completa.
