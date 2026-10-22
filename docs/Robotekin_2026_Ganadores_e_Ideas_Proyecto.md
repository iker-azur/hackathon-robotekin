# Hackathon Robotekin 2026
## Proyectos ganadores de otros hackatones e ideas para nuestro equipo

**Documento de inspiración y selección de proyecto**  
**Actualizado:** 17 de octubre de 2026  
**Equipo:** SO-101 Autonomous Agent · Rubinnova / Rubi I+D

> **Objetivo.** Analizar proyectos premiados en otros hackatones de robótica e identificar una propuesta original, técnicamente defendible y viable para presentar en el Hackathon LeRobot de Robotekin 2026.
>
> **Importante:** los premios y proyectos históricos proceden de las fuentes enlazadas. Las ideas, estimaciones de dificultad y recomendaciones para 2026 son propuestas propias, no anuncios oficiales de Robotekin.

---

## 1. Contexto: ¿qué es el hackatón de Robotekin?

El **Hackathon LeRobot 2026** tendrá lugar del **23 al 25 de octubre de 2026**, en el Robotory de Robotekin, en Vitoria-Gasteiz. La competición está centrada en desarrollar prototipos funcionales de robótica con inteligencia artificial utilizando el ecosistema **LeRobot** de Hugging Face.

**Áreas propuestas por la organización:**

1. **Manipulación de objetos:** recoger, mover, colocar o clasificar piezas.
2. **Visión robótica:** interpretar el entorno mediante cámaras.
3. **Aprendizaje por imitación:** enseñar habilidades al robot mediante demostraciones.
4. **Aplicaciones sociales:** educación, accesibilidad, asistencia e impacto social.

La organización indica que dispondrá de **10 kits SO-101**, además de cámaras, trípodes y ordenadores. Los equipos tienen entre **2 y 5 integrantes**. La entrega de proyectos termina el **domingo 25 a las 13:00**; las presentaciones y premios están programados a partir de las **14:15**. Los premios anunciados son **1.500 €, 900 € y 600 €** para primero, segundo y tercero, respectivamente.

**Referencia oficial:** [Robotekin Hackathon LeRobot 2026](https://hackathon.robotekin.com/).

### Punto de partida de nuestro equipo

Nuestro proyecto previo, **SO-101 Autonomous Agent**, ya persigue el ciclo:

**VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE**

Tenemos trabajo avanzado en simulación, percepción, interfaz web y coordinación entre robots. La oportunidad no consiste necesariamente en inventar otro sistema desde cero, sino en **convertir una parte concreta de lo preparado en una demostración física sólida con el SO-101 real**.

---

## 2. Proyectos premiados en hackatones anteriores

### 2.1. Resumen comparativo

| Evento | Premio o posición | Proyecto | Qué demostró |
|---|---|---|---|
| **LeRobot Bilbao 2025** | 🥇 1.º | **MEGAS XLR** | Bartender con SO-101, aprendizaje por imitación e interacción por voz. |
| LeRobot Bilbao 2025 | 🥈 2.º | **Montaje robótico bimanual** | Dos brazos coordinados para manipular piezas del propio robot. |
| LeRobot Bilbao 2025 | 🥉 3.º | **Tres en raya robótico** | SO-101 entrenado para jugar autónomamente al tres en raya. |
| **LeRobot Worldwide 2025** | 🥇 1.º | **LeChef** | Montaje de sándwiches personalizados con un brazo sobre raíles. |
| LeRobot Worldwide 2025 | 🥈 2.º | **Ez_2_AI** | Manipulación textil para doblar camisetas y prototipo de clasificación de pastillas. |
| LeRobot Worldwide 2025 | 🥉 3.º | **LeSplat / ML6** | Aumento de datos con vistas sintéticas para resistir cambios de cámara. |
| **robo.innovate Múnich 2025** | 🏆 Campeón | **GROWBOT** | Cuidado automatizado de árboles mediante percepción y actuación. |
| robo.innovate Múnich 2025 | 🏆 Más escalable | **ExVo** | Desembalaje automatizado de piezas de automóvil. |
| robo.innovate Múnich 2025 | 🏆 Mejor adecuación problema-solución | **SEAL Robotics** | Propuesta para asegurar contenedores de transporte mediante robótica y visión. |

**Nota:** el evento de Múnich no era un hackatón de LeRobot/SO-101. Se incluye por sus lecciones de aplicación industrial y enfoque de producto. La lista no pretende ser un catálogo exhaustivo de todos los proyectos participantes.

### 2.2. Bilbao 2025 — MEGAS XLR: bartender robótico (1.er premio)

**Idea.** El usuario elegía una bebida en una interfaz web o interactuaba por voz. Un SO-101 ejecutaba movimientos aprendidos para servir una representación de la bebida.

**Aspectos técnicos documentados:**

- Aprendizaje por imitación con una política **ACT**.
- **80 demostraciones**: 50 para una bebida y 30 para otra, según el relato del equipo.
- Entrenamiento con recursos de GPU en la nube.
- Web con un modelo de lenguaje y funciones de voz (STT/TTS).
- Uso de **arena de colores**, en lugar de líquidos, para evitar riesgos para la electrónica.

**Por qué es inspirador.** Resuelve una tarea muy fácil de explicar, tiene un resultado físico visible y crea una experiencia participativa. Su historia también deja una lección: descartaron una funcionalidad adicional porque perjudicaba la fluidez de la demo.

**Qué podemos copiar como estrategia:** una tarea limitada y repetible, una buena narrativa y una interfaz que facilite entender qué se está ordenando al robot.

**Fuentes:** [Relato técnico de un integrante](https://www.elchinaslanli.com/projects/how-we-won-the-lerobot-hackathon-2025-meet-the-ai-bartender) · [Publicación del equipo](https://es.linkedin.com/posts/sergiomoralescobo_ganamos-en-lerobot-bilbao-2025-este-activity-7399009540442079232-yU9k).

### 2.3. Bilbao 2025 — Ensamblaje con dos brazos (2.º premio)

**Idea.** Un sistema con **dos brazos robóticos coordinados** manipulaba piezas impresas en 3D del propio robot, con la visión de automatizar parte de su montaje.

**Aspectos técnicos descritos por sus participantes:**

- Coordinación bimanual.
- Aprendizaje supervisado y por refuerzo.
- Simulación con **Genesis** y adaptación posterior al mundo real.

**Lección.** La colaboración entre robots y la simulación pueden diferenciar un proyecto, pero incrementan la complejidad de integración. Si solo tenemos garantizado un SO-101 físico, no conviene depender de un montaje bimanual para el núcleo de la demostración.

**Fuente:** [Publicación de un integrante del equipo](https://es.linkedin.com/posts/gorka-ayala-madariaga-75571a252_segundo-puesto-en-el-lerobot-hackathon-activity-7399802052878462976-ba9x).

### 2.4. Bilbao 2025 — Tres en raya con SO-101 (3.er premio)

**Idea.** Enseñaron a un brazo SO-101 a colocar fichas y jugar al tres en raya de forma autónoma.

**Lección.** Una tarea conocida por todo el mundo permite entender inmediatamente si el robot está funcionando. No hace falta que el caso de uso sea enorme: importa que la manipulación sea creíble, comprensible y se pueda demostrar.

**Fuente:** [Testimonio y reconocimiento de integrantes del equipo](https://es.linkedin.com/in/joane-carrasco).

### 2.5. Mundial LeRobot 2025 — LeChef (1.er puesto)

**Idea.** Un robot que montaba sándwiches personalizados a partir de instrucciones, incorporando un sistema de desplazamiento lateral para ampliar su alcance.

**Detalles de su dataset:**

- Brazo **SO-100** con raíl lateral.
- **5 tareas** de manipulación de ingredientes.
- **40 episodios por tarea**, **200 episodios** en total.
- Diseño pensado para aprendizaje mediante un modelo **ACT**.

**Lección.** Un comportamiento complejo de alto nivel puede construirse encadenando unas pocas habilidades físicas bien definidas. Es directamente aplicable a un agente que prepara kits de piezas.

**Fuentes:** [Repositorio del dataset LeChef](https://huggingface.co/datasets/LeRobot-worldwide-hackathon/22-leChef-leChef_dataset/blob/main/README.md) · [Crónica de 42AI](https://42-ai.github.io/posts/LeRobotHackathonHighlights/).

### 2.6. Mundial LeRobot 2025 — Ez_2_AI (2.º puesto)

**Idea.** Un equipo del área de San Francisco presentó manipulación de tejidos para doblar camisetas y trabajo experimental en clasificación de pastillas.

**Lección.** Los objetos deformables llaman mucho la atención, pero aumentan los riesgos de agarre, percepción y repetibilidad. Para un hackatón corto, es mejor controlar cuidadosamente materiales y escenarios.

**Fuente:** [Galería de clasificados del Mundial LeRobot](https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners) · [Crónica del evento desde Japón](https://robostadion.com/2025/07/09/lerobotworldwidehackathon/).

### 2.7. Mundial LeRobot 2025 — LeSplat / ML6 (3.er puesto)

**Idea.** En vez de construir una aplicación de manipulación vistosa, el equipo abordó un problema técnico real: una política de imitación puede degradarse cuando se desplaza ligeramente la cámara.

**Solución propuesta.** Reconstruir la escena y generar **nuevas vistas sintéticas** mediante técnicas geométricas y *Gaussian splatting*, para aumentar los datos de entrenamiento y robustecer el modelo.

**Lección.** Una mejora claramente medible de robustez también puede ganar premios. Para nuestro proyecto, un pequeño experimento con desplazamientos de cámara u objetos podría reforzar muchísimo la explicación técnica.

**Fuentes:** [42AI — proyectos premiados](https://42-ai.github.io/posts/LeRobotHackathonHighlights/) · [Informe técnico de ML6](https://www.ml6.eu/en/blog/ai-robotics-a-field-report-on-imitation-learning-with-lerobot).

### 2.8. Múnich 2025 — Tres ganadores industriales

**GROWBOT (campeón):** solución robótica de cuidado de árboles basada en imágenes, medición del suelo y suministro localizado de agua o fertilizante. **Lección:** el impacto práctico se entiende mejor si el robot percibe, decide y actúa sobre un problema real.

**ExVo (premio a la solución más escalable):** propuesta para automatizar el desembalaje de piezas de automóvil, utilizando brazo Franka, visión y detección de puntos de agarre. **Lección:** explicar la escalabilidad y el uso industrial hace más convincente una demo.

**SEAL Robotics (premio a la mejor adecuación problema-solución):** propuesta de automatización para asegurar contenedores de transporte, con herramientas de agarre y visión para localizar elementos de fijación. **Lección:** una necesidad concreta, un flujo sencillo y un mecanismo claro de validación mejoran el relato.

**Fuente oficial:** [Universidad Técnica de Múnich — ganadores de robo.innovate 2025](https://www.mirmi.tum.de/en/mirmi/news/community/article/roboinnovate-hackathon-2025-advancing-robotics-and-ai-solutions/).

### 2.9. Otras ideas destacadas del Mundial (no ganadoras absolutas)

- **LeCyborg (13.º):** tercer brazo portátil, con señales musculares para su control.
- **LeLab (23.º):** interfaz web para facilitar operaciones con LeRobot sin depender tanto del terminal.

Son útiles como referencia para diferenciarse mediante **interacción humana** o **una experiencia de control excelente**, aunque no fueran los proyectos del podio.

**Fuente:** [42AI — resumen de proyectos del Mundial](https://42-ai.github.io/posts/LeRobotHackathonHighlights/).

---

## 3. ¿Qué se repite entre los mejores proyectos?

1. **Una historia sencilla:** «prepara un sándwich», «sirve una bebida» o «recupera una pieza» funciona mejor que describir únicamente modelos y librerías.
2. **Una acción real observable:** el jurado debe poder verificar el resultado físicamente.
3. **Una tecnología con razón de ser:** ACT, percepción o LLM deben resolver un problema identificable, no aparecer solo por moda.
4. **Un alcance acotado:** entrenar tres acciones fiables suele ser mejor que prometer veinte que fallan.
5. **Una demostración preparada para fallos:** recuperación, reinicio, límites de seguridad y vídeo de respaldo.
6. **Evidencias:** tasa de éxito, número de demostraciones, latencia y condiciones de prueba.

**Nuestra oportunidad diferencial:** pasar de un simple *pick and place* a un sistema que **observa, planifica, ejecuta, verifica y se recupera**.

---

## 4. Cinco ideas propuestas para Robotekin 2026

> Las siguientes ideas **no son proyectos premiados**: son alternativas nuevas para nuestro equipo. La dificultad y el potencial son valoraciones cualitativas propias.

### Idea 1 — RecoveryBot: un robot que se recupera de errores

**Concepto.** Una pequeña célula de montaje o preparación de kits. El usuario pide colocar piezas en destinos determinados. Si una pieza cambia de posición o el agarre falla, el robot detecta el problema y decide qué hacer.

**Demo ante el jurado:**

1. Pedimos: «Prepara un kit con una pieza roja y dos azules».
2. El robot identifica y coloca la primera pieza.
3. Movemos una pieza restante a otra posición dentro de su zona de alcance.
4. El sistema vuelve a observar la escena, actualiza el plan y continúa.
5. Una verificación final comprueba el contenido del kit.

**Tecnologías posibles:** cámara cenital, detección de objetos, calibración cámara-mesa, política de manipulación con LeRobot y máquina de estados para gestionar recuperaciones.

**MVP realista:** una pieza de cada color, una caja y recuperación de un único fallo conocido. **Versión avanzada:** reintentos de agarre y corrección de piezas mal colocadas.

**Principal riesgo:** confundir «volver a ejecutar una trayectoria» con recuperarse realmente. La recuperación debe utilizar **una observación nueva** y una decisión basada en ella.

**Valoración:** dificultad **4/5** · impacto de demo **5/5**.

### Idea 2 — CircuitBot: construir un circuito que enciende un LED

**Concepto.** El robot monta bloques electrónicos grandes, seguros y con conectores tolerantes. La tarea termina cuando se enciende un LED real.

**Demo:** «Monta un circuito para encender la luz» → reconoce bloques → planifica orden → coloca batería simulada o fuente segura, interruptor y LED → verifica encendido.

**MVP:** colocar 2–3 módulos con contactos magnéticos y posiciones de montaje definidas. **Versión avanzada:** circuitos alternativos.

**Principal riesgo:** precisión mecánica y falsos contactos. Diseñar las uniones para perdonar unos milímetros de error es esencial.

**Valoración:** dificultad **4/5** · impacto de demo **5/5**.

### Idea 3 — TeachBack: corregir y mejorar una política robótica

**Concepto.** El robot intenta una tarea, un humano corrige una ejecución con teleoperación y esa corrección entra en el flujo de datos para entrenar una política mejor.

**Demo:** comparar la versión inicial con una versión mejorada y mostrar ejemplos de intervención humana.

**MVP:** dos políticas entrenadas de antemano sobre datasets distintos; comparar tasas de éxito con el mismo escenario. **Versión avanzada:** recoger una corrección delante del jurado y dejarla lista para el siguiente entrenamiento.

**Principal riesgo:** prometer aprendizaje instantáneo sin disponer de tiempo de GPU ni una evaluación rigurosa. **No se debe afirmar que aprende en directo si solo se está guardando una corrección.**

**Valoración:** dificultad **5/5** · interés técnico **5/5**.

### Idea 4 — Prompt-to-Pixel: convertir instrucciones en arte físico

**Concepto.** El usuario describe una figura sencilla y el robot la construye con fichas de colores sobre una cuadrícula.

**Demo:** «Haz un corazón» o «Haz una composición roja y verde» → se genera un patrón → el SO-101 coloca las fichas → se verifica la imagen resultante.

**MVP:** cuadrícula de 2 × 3, pocas piezas rígidas y 2–3 patrones conocidos. **Versión avanzada:** patrones generados dinámicamente y corrección de casillas erróneas.

**Principal riesgo:** demasiadas manipulaciones seguidas aumentan tiempo y probabilidad acumulada de error.

**Valoración:** dificultad **3/5** · impacto visual **5/5**.

### Idea 5 — ReUseBot: recuperar componentes de un producto simulado

**Concepto.** Una estación de desmontaje y clasificación para piezas reutilizables de un dispositivo de demostración.

**Demo:** identifica un módulo recuperable, lo extrae de un alojamiento accesible y lo deja en la bandeja correcta; muestra después el resultado del proceso.

**MVP:** carcasa abierta, piezas grandes de extracción sencilla y materiales inertes. **Versión avanzada:** inspección de varios tipos de módulo y priorización de recuperación.

**Principal riesgo:** desmontar de verdad puede exigir fuerzas, herramientas y precisión incompatibles con un SO-101. Mejor una maqueta diseñada para manipulación segura. **No utilizar baterías reales dañadas ni circuitos energizados.**

**Valoración:** dificultad **4/5** · aplicación industrial **5/5**.

### Dos variantes especialmente compatibles con nuestro trabajo previo

- **KitBuilder AI:** preparar cajas personalizadas a partir de instrucciones en lenguaje natural. Es la opción de menor riesgo y puede convertirse en la base de RecoveryBot.
- **MultiRobot Assembly:** dos SO-101 que se reparten tareas de clasificación y montaje. Puede ser una evolución espectacular, pero solo si la organización confirma el acceso y tiempo de uso de dos robots físicos.

---

## 5. Comparativa para elegir proyecto

Escala de **1 (bajo) a 5 (alto)**. Son estimaciones orientativas, no mediciones ni puntuaciones oficiales.

| Idea | Impacto visual | Viabilidad en 48 h | Diferenciación | Riesgo técnico |
|---|:---:|:---:|:---:|:---:|
| **RecoveryBot** | 5 | 4 | 5 | 4 |
| **CircuitBot** | 5 | 3 | 5 | 4 |
| **TeachBack** | 4 | 2 | 5 | 5 |
| **Prompt-to-Pixel** | 5 | 5 | 4 | 3 |
| **ReUseBot** | 4 | 4 | 4 | 4 |

**Recomendación principal: RecoveryBot + KitBuilder.** Combina un resultado fácil de entender con una diferenciación técnica visible: el robot no solo ejecuta una instrucción, sino que **comprueba lo ocurrido y modifica su comportamiento cuando algo cambia**.

**Plan B más seguro:** Prompt-to-Pixel con pocos patrones y una tasa de éxito elevada.

**Opción de mayor espectáculo, pero más exigente mecánicamente:** CircuitBot.

---

## 6. Propuesta concreta: SO-101 Autonomous Agent — RecoveryKit

### Historia de la demo

> **Persona:** «Prepara una caja con una pieza roja y dos azules».
>
> **Robot:** detecta las piezas, planifica tres recogidas, coloca y verifica.
>
> **Perturbación:** durante la prueba desplazamos una pieza que el robot todavía no ha recogido.
>
> **Recuperación:** el sistema detecta el cambio, recalcula el objetivo de recogida y continúa.

### Pipeline del sistema

```text
PETICIÓN DEL USUARIO
        ↓
ENTENDER → convertir a lista de tareas
        ↓
VER → detectar piezas, cajas y posiciones
        ↓
PLANIFICAR → elegir orden de recogida
        ↓
ACTUAR → ejecutar habilidad de pick & place
        ↓
VERIFICAR → comparar resultado con objetivo
        ↓
¿RESULTADO CORRECTO?
    ├── Sí → siguiente tarea / finalizar
    └── No → identificar causa, reobservar y recuperar
```

### Qué enseñar en pantalla

- **Objetivo actual:** «Kit: 1 roja + 2 azules».
- **Cámara:** observación principal con detecciones legibles.
- **Lista de tareas:** pendiente / en curso / completada / fallida.
- **Estado de recuperación:** «Pieza desplazada → nueva localización detectada → reintento».
- **Indicadores medidos:** número de piezas correctas, tiempo de ejecución, reintentos y resultado final.

La interfaz debe ser **clara, en español y rápida**. Evitar paneles excesivos o una visualización artificial de razonamientos que no corresponda a decisiones reales del software.

### Alcance mínimo y ampliaciones

| Nivel | Qué incluye | Cuándo darlo por terminado |
|---|---|---|
| **MVP** | Detectar dos colores, recoger y depositar una pieza, verificar con cámara. | Repetible en el robot físico. |
| **Demo competitiva** | Preparar un kit de 2–3 piezas y recuperar un desplazamiento intencionado. | Funciona en varias pruebas consecutivas. |
| **Ampliación** | Reintentar agarres fallidos, varios destinos y dos robots colaborativos. | Solo tras estabilizar los niveles previos. |

### Principios técnicos

- **Calibrar primero:** definir origen, escala, límites y zona segura de la mesa.
- **No depender de coordenadas mágicas:** convertir detecciones de cámara en posiciones de trabajo mediante calibración.
- **Evitar colisiones:** trayectorias seguras y parada disponible.
- **Controlar el entorno:** iluminación, tamaño de piezas, agarre y fondo suficientemente estables.
- **Usar aprendizaje por imitación donde aporte valor:** demostrar una habilidad entrenada, sin presentar todo el sistema como aprendizaje si parte es planificación determinista.
- **No depender de conectividad para el gesto crítico:** si LLM o servicios en la nube fallan, ejecutar una instrucción local de respaldo.
- **Ensayar una perturbación concreta:** que siempre pueda colocarse dentro del área alcanzable y segura.

---

## 7. Cómo demostrar que el proyecto funciona

En lugar de prometer un porcentaje arbitrario, medir:

1. **20 ejecuciones normales** del escenario elegido.
2. **20 ejecuciones con perturbación**, manteniendo condiciones comparables.
3. **Tasa de éxito completa:** kits terminados correctamente / intentos.
4. **Tasa de recuperación:** incidencias recuperadas / incidencias detectadas.
5. **Tiempo total de ejecución:** mediana y dispersión, no solo el mejor caso.
6. **Errores frecuentes:** pieza no detectada, mal agarre, depósito incorrecto o pérdida de calibración.

Estos números son una **propuesta de protocolo de evaluación**, no resultados obtenidos todavía.

**Qué enseñar al jurado:** una demostración real, la explicación del fallo provocado y una tabla corta con los resultados medidos. Si existe un fallo durante la presentación, activar un procedimiento de reinicio conocido.

---

## 8. Plan de trabajo durante el hackatón

| Momento | Prioridad | Entregable |
|---|---|---|
| **Viernes 23, tarde/noche** | Montar, calibrar, probar cámara y agarre. | Una recogida y un depósito fiables. |
| **Sábado 24, mañana** | Capturar demostraciones y entrenar una habilidad. | Política evaluable o movimiento alternativo seguro. |
| **Sábado 24, tarde** | Integrar percepción, planificador y verificación. | Preparación de un kit completo. |
| **Sábado 24, noche** | Introducir y probar recuperación; medir resultados. | Demostración estable y registros de pruebas. |
| **Domingo 25, mañana** | Congelar funcionalidades, ensayar y preparar presentación. | Demo, resultados, repositorio y vídeo de respaldo. |
| **Domingo 25, antes de las 13:00** | Entregar según los requisitos de la organización. | Proyecto presentado dentro de plazo. |

**Regla de prioridad:** primero hardware fiable, luego habilidad física, después percepción y verificación; en último lugar añadir voz, LLM o efectos visuales.

---

## 9. Conclusión

Los resultados de Bilbao y del Mundial de LeRobot sugieren que un buen proyecto combina **una tarea humana reconocible, robótica real y una ejecución clara**. Múnich añade otra lección: conectar la demo con una necesidad industrial concreta ayuda a justificar su valor.

Para este equipo, **RecoveryKit** aprovecha lo trabajado en simulación y transmite una idea potente:

> **No queremos un robot que repita movimientos. Queremos un robot que observe, comprenda una tarea, actúe y sepa qué hacer cuando el mundo no se comporta como esperaba.**

El objetivo no debe ser aparentar autonomía ilimitada, sino **demostrar una autonomía acotada, verificable y robusta** en el tiempo real disponible.

---

## 10. Fuentes y referencias

### Hackathon Robotekin 2026

1. [Web oficial de Robotekin — programa, áreas, recursos y premios](https://hackathon.robotekin.com/).
2. [Robotekin — noticia de la convocatoria 2026](https://robotekin.com/robotekinek-arabako-lehen-hackathona-antolatu-du-50-gaztek-adimen-artifizialaren-bidez-robotak-trebatzeko/).

### LeRobot Bilbao 2025

3. [Web del hackatón de Bilbao 2025](https://hackathonbilbao.eus/).
4. [MEGAS XLR — relato técnico del equipo ganador](https://www.elchinaslanli.com/projects/how-we-won-the-lerobot-hackathon-2025-meet-the-ai-bartender).
5. [MEGAS XLR — explicación del equipo](https://es.linkedin.com/posts/sergiomoralescobo_ganamos-en-lerobot-bilbao-2025-este-activity-7399009540442079232-yU9k).
6. [Segundo premio — publicación de un participante](https://es.linkedin.com/posts/gorka-ayala-madariaga-75571a252_segundo-puesto-en-el-lerobot-hackathon-activity-7399802052878462976-ba9x).
7. [Tercer premio — experiencia y reconocimiento del equipo de tres en raya](https://es.linkedin.com/in/joane-carrasco).

### LeRobot Worldwide Hackathon 2025

8. [Galería de los 30 proyectos premiados](https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners).
9. [Archivos de vídeos ordenados por posición](https://huggingface.co/datasets/maringetxway/all-winners/tree/main).
10. [LeChef — dataset y tareas](https://huggingface.co/datasets/LeRobot-worldwide-hackathon/22-leChef-leChef_dataset/blob/main/README.md).
11. [42AI París — explicación de LeChef, LeSplat, LeCyborg y LeLab](https://42-ai.github.io/posts/LeRobotHackathonHighlights/).
12. [ML6 — informe de aprendizaje por imitación y robustez de cámaras](https://www.ml6.eu/en/blog/ai-robotics-a-field-report-on-imitation-learning-with-lerobot).
13. [Crónica de un anfitrión del Mundial desde Tokio](https://robostadion.com/2025/07/09/lerobotworldwidehackathon/).

### robo.innovate Múnich 2025

14. [Universidad Técnica de Múnich — resultados oficiales](https://www.mirmi.tum.de/en/mirmi/news/community/article/roboinnovate-hackathon-2025-advancing-robotics-and-ai-solutions/).

### Herramientas y documentación de referencia

15. [Repositorio oficial de LeRobot](https://github.com/huggingface/lerobot).
16. [Documentación oficial de LeRobot](https://huggingface.co/docs/lerobot/index).

---

*Documento de trabajo para discusión interna del equipo. Los objetivos y valoraciones son orientativos; las capacidades reales deberán verificarse con el hardware disponible durante el hackatón.*
