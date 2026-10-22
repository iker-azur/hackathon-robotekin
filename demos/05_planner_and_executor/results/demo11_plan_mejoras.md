# Demo 11 — Plan de mejora implementable

**Proyecto:** SO-101 Autonomous Agent · Robotekin Hackathon 2026  
**Estado:** análisis y hoja de ruta; implementación parcial de P0/P1 aplicada en los módulos indicados.  
**Entorno de referencia observado:** Windows, Isaac Sim 5.1, Isaac Lab/LeIsaac, RTX 5060 Ti (16 GB), 32 GB RAM.  
**Nota:** cifras históricas citadas aquí provienen de informes del repositorio; no son una nueva campaña ejecutada durante esta implementación.

## 1. Resumen ejecutivo

Demo 11 ya tiene una cadena instrumentada: escena → cámara RGB-D → segmentación HSV/componentes → localización en plano de mesa → asociación con catálogo → plan determinista → servo y agarre físico → verificación por poses de Isaac. El problema inmediato no es la ausencia de un modelo más grande: son interfaces frágiles entre etapas, incertidumbre de localización, agarre/contacto de la bola y ejecución lenta. La web ya muestra mapa y lista, pero no permitía editar visualmente y guardar varias escenas.

La recomendación es mantener OpenCV y la cámara cenital, usar el catálogo conocido para resolver tipos y dimensiones, conservar planificación y verificación deterministas, y mejorar medición, confianza y recuperación. No se recomienda YOLO ni entrenar un JEPA ahora. No se encontró implementación JEPA en Demo 11 y, para una escena controlada de objetos de colores saturados, una red añade modelo, datos y carga CUDA sin resolver por sí misma calibración, IK ni contacto.

**Cambios aplicados en esta implementación:** `yaw_deg` en esquema y simulación; AABB girada para validar espacio/colisión; asociación por catálogo que no descarta por tipo visual y adopta geometría declarada; pausa/reanudación entre pasos de física; mapa arrastrable; formulario de X/Y en mm, giro y tamaño para piezas individuales; escenas guardadas por nombre con endpoints de carga/guardado.

## 2. Arquitectura actual y problemas

| Etapa | Módulos | Responsabilidad observada |
|---|---|---|
| Entorno/escena | `scene_config.py`, `scene_spec.py`, `real_world.py`, `11_real_sorting_with_monitor.py`, `hollow.py`, `pen.py` | Configuración versionada; cubos/bolas y recipientes; medidas, materiales, alcance y factibilidad de pinza. |
| Cámara | `isaac_io.py`, `scene_spec.camera_poses()` | Pinhole cenital y variantes, RGB + depth de plano, 640×480, `update_period=0`; intrínsecos y pose de Isaac. El monitor usa una ruta ligera sin depth. |
| Detección | `colors.py`, `detector.py`, `perceive.py` | Bandas HSV, máscara de mesa/brazo, blobs/contornos, forma/color/centro; profundidad mediana y huella cuando disponible. |
| Mundo/asociación | `MonitoredRobot.look/link/locate`, catálogo | Asignación húngara por color, tipo y distancia a seguimiento anterior; inventario esperado en escena. |
| Intención/plan | `command_model_v2/`, `intent_parser.py`, `ground.py`, `planner.py`, `capacity.py` | Orden española → entidades/destino → poses y huecos no solapados. |
| Acción/verificación | `StrictExecutor`, `executor.py`, `TaskLedger` | IK/acciones LeIsaac y contacto; aproximación, cierre, traslado y soltado; verificación de cavidad por pose del simulador, reintento limitado. |
| Frontend/telemetría | `monitor/server.py`, `monitor/static/index.html`, `telemetry.py`, `supervisor.py`, `frame_encoder.py` | FastAPI/WebSocket + REST, supervisor separado del proceso Isaac, estado, JPEG y cola de órdenes. |

Flujo de extremo a extremo: navegador → API/bus → worker Isaac → escena/cámara → percepción → enlace al catálogo → planner → IK/actuadores/física → verificación → telemetría y fotogramas. Es una separación útil: Isaac conserva su hilo principal y la web no bloquea la simulación.

### Hallazgos y certeza

- **Confirmado por código e informes:** proyección de cámara y convenio de ejes fueron contrastados con blobs; la detección robusta tuvo 4,16 mm de error XY medio en el fixture guardado y ~16,6 ms de latencia media. `demo11_metrics.json` registra 90/92 instancias asociadas y 87 tipos correctos en robusto frente a 88 asociadas/64 tipos correctos en legado. Son fixtures, no una tasa de despliegue.
- **Histórico de manipulación:** una corrida de preset de dos cubos terminó 2/2 sin rechazo; corridas con bola tuvieron fallos/rechazos. El paso de física se midió en 33,03 ms (0,505× tiempo real); una pieza tardaba ~33 s. La cámara ~30 ms no es el cuello dominante del tiempo de tarea.
- **Aún sin medida reciente:** error P50/P95 por forma/luz/pose, timestamps/edad de frame, sincronización de percepción durante el movimiento y tasa de agarre luego de los cambios actuales.
- Cámara: resolución 640×480; percepción captura RGB y depth con actualización bajo demanda. El buffer depth implica copia GPU→CPU y conversión float64; `capture_rgb` evita esa rama para video. Monitor limitado por `--cam-fps` (3 fps por defecto). No hay tasa fija documentada de adquisición para cada mirada.

### Causas raíz de detección y fallos de bola

1. HSV y contornos son rápidos y adecuados para colores contrastados; reflejos, sombra, autoexposición, oclusión o blobs fusionados degradan máscara/forma. No es automáticamente un problema que YOLO resuelva.
2. El centroide visual puede diferir del centro de apoyo por perspectiva/oclusiones. La depth puede mezclar tabla y objeto en bordes; el centro de pinza debe ser evaluado por huella y forma.
3. Proyección al plano depende de calibración y altura asumida; depth entrega superficie, no identidad semántica. La asociación temporal limita saltos, pero objetos idénticos cercanos siguen siendo ambiguos.
4. El enlazado exigía antes que la forma estimada coincidiera exactamente con el tipo declarado. Se podía perder una entidad conocida por una confusión de contorno. El cambio de esta entrega permite asociar por color, clase objeto/recipiente y cercanía, y luego usar el tipo/dimensiones del catálogo. Esto no repara una máscara ausente ni resuelve oclusiones.
5. La bola carece de caras que limiten el movimiento: una pinza descentrada la empuja/rueda; si su ancho supera apertura útil o falta fricción/fuerza, corregir píxeles no ayuda. El código ya calibra esfera de 30 mm, cierre y asentamiento, servo de centrado y agarre estricto. No bajar los pasos de settle de cierre sin experimento.

Diagnóstico por síntomas: sin blob/segmentos unidos → luz/máscara; blob con XY malo → calibración/profundidad/centro; movimiento al cerrar → contacto y estrategia; cierra pero no eleva → ancho/fuerza/fricción/altura; cae al transportar → retención/inercia; falla en bandeja → margen/altura/cavidad/retirada. Verificación por pose es un oráculo de simulación, no una observación externa.

## 3. Editor de entorno y validación

El modal se amplió y adapta su layout a móvil. El plano cenital usa workspace, brazo y alcance del servidor; arrastrar desplaza pieza/recipiente; piezas individuales exponen X/Y en mm, giro y tamaño; arrastrar un grupo lo descompone en entradas individuales conservando posición de catálogo. Los endpoints `GET /api/scenes`, `GET /api/scenes/{name}`, `POST /api/scenes/{name}` guardan JSON validado en `out11/scenes/`; nombre limitado a caracteres seguros; se rechaza guardar con simulación activa. X/Y/yaw se aplican a cuaternión wxyz al instanciar y recolocar.

**Capacidad vigente:** cubo/cuboide, bola/esfera, cilindro (experimental, sin campaña de agarre), colores admitidos y recipientes rectangulares huecos. El catalogo declara forma/dimensiones; color y posicion visual asocian la deteccion con la entidad. `scene_config.py` valida alcance aproximado, huella girada conservadora y agarre por apertura. No representa colision completa del brazo ni prueba trayectoria IK.

**Estado del editor:** tabla cenital con drag, dimensiones/giro de piezas y recipientes, biblioteca, JSON, undo/redo, reset, validacion backend debounced y aviso de entidad invalida. Se bloquea al ejecutar. **Pendiente:** preflight IK/collision checking, triangulos/prismas con malla y trial fisico, y comprobacion manual de layout/teclado en viewport movil.

## 4. Comparación de percepción

| Opción | Precisión/latencia/GPU | Datos y complejidad | Recomendación |
|---|---|---|---|
| HSV + blobs/contornos | Buena en colores saturados separados; muy baja latencia CPU y ya instrumentada | Sin entrenamiento; calibrar bandas y máscara | Detector principal a corto plazo. |
| YOLO detección | Mejor semántica si existe variedad suficiente; caja no da centro métrico, comparte GPU | Datos de poses/luz/modelos, pesos, integración y tuning | No añadir ahora; no resuelve geometría de agarre. |
| YOLO-seg | Máscara por instancia puede ayudar con oclusión/segmentos fusionados | Etiquetas de máscaras o sintéticos y prueba de domain gap; inferencia adicional | Solo como experimento si benchmark prueba que HSV incumple recall/XY. |
| RGB-D | Depth/plano mejora altura/huella; copia GPU→CPU cuesta tiempo | Ya disponible; comprobar NaN, borde y convenciones | Complemento actual a blobs; conservar máscara interior robusta. |
| Ground truth Isaac | Excelente para evaluación/etiquetas, no para percepción del producto | No requiere entrenamiento; riesgo de “oracle” | Solo verdad de referencia/generación de datos. |
| Híbrido catálogo + CV + depth + tracking | Mejor relación costo/fiabilidad para escena acotada; ambiguo si se solapan gemelos | Sin modelo nuevo; medir incertidumbre y seguimiento | **Recomendado.** |

La ruta propuesta es OpenCV detecta color/huella, depth/plano estima XYZ, catálogo declara tipo/tamaño, asociación temporal resuelve identidad y devuelve confianza/ambigüedad. Registrar timestamp desde captura a servo para descartar frames viejos.

### Auditoría de la visión actual y trabajo pendiente (revisión 2026-10-18)

La base actual ya es más capaz que una segmentación de blobs sencilla: `perceive.py` usa máscaras HSV por color, ROI de espacio de trabajo, componentes conectados, separación de manchas por picos de transformada de distancia y reparto por profundidad cuando una pieza toca visualmente un recipiente. Clasifica formas con momentos/contornos y profundidad; `link()` asocia las detecciones al catálogo con Hungarian y cercanía a la última posición conocida. La cámara Isaac publica RGB y profundidad, y el panel usa RGB ligero para el vídeo. Estos mecanismos deben medirse y afinarse antes de añadir una red neuronal.

La principal limitación para objetos muy próximos no es únicamente que “el detector no vea la forma”: cuando dos piezas del mismo color se tocan, la máscara inicial se convierte en una sola región. `split_touching()` solo divide grupos compactos si encuentra picos suficientemente separados y si las áreas resultantes pasan varios filtros. Conservadoramente devuelve la mancha original si el clump es alargado/grande, un pico pequeño queda oculto por otro, la división genera fragmentos muy desiguales o no conserva área. En esos casos se pierde el recuento de instancias o el centro individual, aunque el contorno aún parezca razonable. La segmentación por colores diferentes ayuda con objetos de distinto color, pero bandas HSV que se solapan pueden dar duplicados o asignaciones ambiguas.

Hallazgos que deben guiar la siguiente iteración:

- **Instancias juntas:** el splitter geométrico existe, pero sus heurísticas no equivalen a una garantía para dos objetos parcialmente solapados, tamaños distintos oclusos, formas no convexas y agrupaciones de más de dos. La cifra histórica de `demo11_metrics.json` (90/92 emparejadas y 87 tipos correctos en 18 fotogramas robustos) no es un benchmark suficiente para estos casos: mezcla fixtures de calibración y variantes oscurecidas, y no estratifica por separación/oclusión. No debe presentarse como precisión en cualquier escena.
- **Forma:** bolas, cubos y prismas pueden tener siluetas parecidas por perspectiva, poca resolución oclusión y sombras. El clasificador debe poder responder “tipo ambiguo”; en Demo 11 el catálogo suele proporcionar el tipo real una vez asociada la mancha, por lo que una clase visual equivocada no debe descartar una instancia conocida.
- **Localización:** `pos_m` parte del centroide y la intersección del rayo con el plano de mesa; la profundidad añade medidas de superficie/huella y correcciones específicas, pero no debe mezclarse como si fuera automáticamente el centro de agarre. Bordes de máscara incluyen tabla, suelo de bandeja o pared; una mediana simple puede sesgar altura y centro. Para bolas y objetos altos conviene estimar centro de apoyo/centro de agarre usando profundidad válida y geometría del catálogo, y publicar su incertidumbre.
- **Identidad y sincronía:** la asignación húngara usa color, posición previa y tolerancia; es vulnerable a piezas casi idénticas que se cruzan, a una división/duplicación entre fotogramas y a saltos durante el contacto. La percepción para planificar debe ser una observación fresca posterior al asentamiento, con timestamp, no el último JPEG publicado por el monitor. El vídeo de baja frecuencia es para supervisión y no constituye por sí solo seguimiento visual continuo.
- **Confianza:** `confidence_of()` es un score heurístico de calidad, no una probabilidad calibrada. No se debe usar un umbral de “confianza 0.8” como garantía hasta medir calibración y tasa de abstención por categoría.

Orden recomendado de mejora: (1) guardar fotogramas RGB-D, verdad de simulador y parámetros por semilla; (2) benchmark por color, forma, tamaño, distancia entre centros, solape, iluminación y cámara; (3) mejorar la separación de instancias usando, en este orden, componentes por color, discontinuidades de profundidad, transformada de distancia con semillas guiadas por número/tamaño del catálogo y contornos/huellas; (4) asociar observaciones consecutivas con timestamp, gating espacial y una pista explícita por instancia; (5) calibrar error/abstención y corrección de centro de agarre. Si tras ello el recall de grupos cercanos continúa por debajo del objetivo, comparar YOLO-seg con OpenCV sobre exactamente el mismo conjunto y el mismo hardware. La red sería un candidato a generar máscaras de instancia, no una autoridad de posición: RGB-D/calibración y catálogo aún determinan pose y geometría.

**No seleccionar YOLO todavía.** El escenario tiene un conjunto pequeño de colores y geometrías conocidos, segmentación CPU rápida y profundidad de Isaac; el coste principal medido de la tarea está en los pasos de física, no en percepción. YOLO-seg añade pesos, instalación/carga en el mismo proceso que PyTorch/Isaac, GPU compartida, datos con máscaras y posible diferencia entre render y banco real. El experimento se justifica si el benchmark demuestra que OpenCV no separa ocluidos/tocantes en el rango requerido. Los datos sintéticos de Isaac pueden aportar etiquetas de instancia/depth, pero se separan por escena/semilla y la evaluación final se hace con semillas no vistas y fotogramas reales del banco cuando existan.

## 5. Manipulación

- Representar objetivo con identidad, centro, altura, yaw, dimensiones, confianza/covarianza y timestamp. No iniciar descenso con observación débil.
- Exigir observaciones consistentes; abortar/reobservar si el objeto se mueve en aproximación.
- Cubo: agarre paralelo a cara, pregrasp 30–50 mm arriba, avance/cierre medidos, ascenso de prueba.
- Bola: acercamiento superior centrado, cierre calibrado y lift test de 5–10 mm; comprobar que sigue con el efector antes del traslado. Mantener bolas de 30 mm dentro de zona medida.
- Recipiente: approach vertical, margen interior respecto a pared y retirada vertical.
- Verificar tras cierre, ascenso y colocación. Usar eventos de contacto/pose de simulador como verificación de simulación y, si se despliega hardware, añadir observación física independiente.
- Recuperación acotada: abrir/retirarse, observar y recentrar una vez; tras repetición reportar causa y parar el objetivo. Evitar empujar repetidamente o aumentar fuerza sin evidencia.

La pausa implementada espera en el borde entre pasos de física, conserva pose y reanuda; stop limpia la pausa. No interrumpe una llamada `env.step` ya iniciada.

## 6. JEPA/world model

La búsqueda en el directorio de Demo 11 no encuentra implementación, checkpoint, dataset ni proceso de entrenamiento JEPA. “Real-world model” en documentación se refiere a modelo de materiales/física, no a Joint Embedding Predictive Architecture. El planificador que controla el robot es explícito; el resultado se verifica con pose de Isaac. No se puede atribuir una falla de agarre a un modelo que no forma parte de esta ruta.

| Opción | Decisión |
|---|---|
| Mantener actual | Sí: no hay modelo JEPA desplegado que modificar. |
| Fine-tune/retrain | No justificado sin baseline/modelo/objetivo. 16 GB VRAM permite explorar un encoder pequeño, pero no arreglará IK/contacto automáticamente. |
| Cambiar rol | Solo predictor de riesgo/resultado, nunca sustituto de geometría, límites o verificación. |
| Híbrido explícito+aprendido | Catálogo + IK/plan determinista para seguridad; añadir predictor si se mide un fallo recurrente fuera del modelo explícito. |

Gate futuro: ≥1000 transiciones con pose/acción/RGB-D/contacto/éxito, separación train/val por escena y semilla; medir Brier/AUROC y mejora de tarea real frente a baseline. Exigir ≥5 puntos porcentuales de éxito adicional en ≥200 pruebas sin latencia por orden >10%; si no, no desplegar. Empezar por dataset y benchmark, no por reentrenamiento a ciegas.

## 7. Frontend y experiencia

Se recomienda mantener el modal por ahora: panel local y ajustes de sesión conviven con un mapa 2D que permite posiciones métricas con menos complejidad que una vista Three.js. En desktop el editor ampliado distribuye lista y recipientes; en móvil se apila. Una vista dedicada solo se justifica al comprobar que usuarios deban manejar escenas complejas o el modal no cabe.

Mostrar arriba estado/pausa/stop; centro con cámara anotada, frescura del frame y detecciones; al lado plan actual, pieza/destino y acciones; al final, motivo de rechazo/recuperación y veredicto. No saturar con datos internos repetidos. Pendiente: overlay de bbox/centro/confianza, edad del frame, reset/undo, foco teclado, contraste, errores accionables y estado de conexión. Todo el texto es español.

## 8. Roadmap priorizado

Estimaciones para una persona; Isaac/Windows puede añadir tiempo de instalación y ensayo.

### P0 — Fiabilidad (parcial)

**Aplicado:** quitar veto de tipo visual al asociar por escena; usar tipo/tamaño declarado; pausa segura entre pasos. En la continuación actual, el splitter usa también el tamaño y la huella máxima de la escena; una mancha grande que no se logra dividir se marca como ambigua, queda fuera de asociación/grounding y fuerza una vista de respaldo cuando existe. Se añade timestamp de captura y no se permite localizar una pieza para agarre con una observación de más de 1 s. El panel diferencia grupos por reobservar y los excluye del vocabulario confirmado. El benchmark offline ahora puntúa configuración de producción, abstenciones, precisión/recall y P95. **Pendiente:** banco RGB-D por semillas con clusters controlados; tracking temporal persistente y calibración de confianza/XY; CSV de fallos por etapa; calibrar centro de contacto y verificar recuperación.

**Visión, prioridad dentro de P0:** el splitter de manchas y el enlace por catálogo ya están implementados; no se considera cerrado el reconocimiento en grupos. Añadir conjunto de regresión RGB-D etiquetado con instancias y escenarios emparejados (separados, casi tocándose, tocándose, solape parcial, tamaños mezclados y oclusión por pinza); extraer recall/precision por instancia y color, error XY P50/P95, conteo de divisiones/duplicados y latencia. Después, hacer la segmentación sensible a profundidad y a los tamaños/clases esperados; mantener detección ambigua como resultado válido y solicitar otra vista cuando no haya evidencia suficiente. Añadir tracking temporal con IDs/edad de observación y usar únicamente capturas frescas para iniciar/reanclar un agarre. Comparar YOLO-seg solo como prueba controlada si estos cambios no alcanzan recall.

- **Archivos probables:** `perceive.py` (máscaras, splitting, score), `scene.py` (observaciones/timestamps), `isaac_io.py` (captura sincronizada y metadatos), `11_real_sorting_with_monitor.py` (frescura, enlace y reobservación), `monitor/static/index.html` (overlay de instancias/confianza/ambigüedad) y fixtures/herramientas en `tests/` y `tools/`.
- **Esfuerzo orientativo:** 3–5 días para el benchmark y mejoras OpenCV/tracking; 2–4 días adicionales para una comparación acotada de YOLO-seg si los resultados la justifican. Depende de capturar verdad RGB-D reproducible y no incluye entrenamiento extenso.
- **Aceptación de visión:** ≥98% de recall de instancia en escenas separadas; ≥92% en grupos cercanos donde cada pieza conserva evidencia visual suficiente; error XY P95 ≤6 mm en la cámara cenital calibrada y ≤10 mm en vistas oblicuas ensayadas; ≤30 ms P95 de percepción CPU a resolución de producción; 0 agarres iniciados con observaciones caducadas o ambiguas. Informar por separado casos totalmente ocluidos o sin evidencia separable, que deben producir abstención/reobservación y nunca una detección inventada.

- **Archivos:** `perceive.py`, `scene.py`, `11_real_sorting_with_monitor.py`, `executor.py`, `telemetry.py`, monitor y `tools/verify_demo11_session.py`.
- **Esfuerzo:** 4–7 días entre instrumentación, ajuste y ensayos. Riesgo: cambiar fricción/pasos reduce éxito aunque mejore tiempo.
- **Aceptación:** recall ≥95% con objetos visibles; P95 XY ≤8 mm en zona ensayada; 30/40 picks de cubo, ≥24/40 bola; ≥90% de fallos detectados y clasificados dentro de una observación.

Los umbrales de visión anteriores se validan también con grupos cercanos; el recall agregado no puede ocultar una clase difícil. Si OpenCV los cumple, mantenerlo como detector de producción y no introducir YOLO.

### P1 — Editor (parcial)

**Aplicado:** drag, X/Y, escala y giro de objetos individuales; dimensiones de recipiente; biblioteca, importar/exportar, reset y undo/redo; validacion backend continua; resaltado de entidades invalidas; lock de editor durante ejecucion. **Pendiente:** preflight de IK/colision y campaña aleatoria repetible.

- **Archivos:** `scene_config.py`, `scene_spec.py`, `monitor/server.py`, `monitor/static/index.html`, builder Isaac.
- **Esfuerzo restante:** 2–4 días. Riesgo: alcance del efector no demuestra trayectoria libre; AABB puede ser conservadora.
- **Aceptación:** escena inválida rechazada antes de iniciar; 100 posiciones aleatorias no pasan fuera de límites/solape; round-trip escena→JSON→catálogo idéntico; respuesta backend <250 ms.

### P2 — Geometrías nuevas

**Pendiente:** soporte integral de forma física, masa, segmentación, interacción natural y agarre; no exponer solo un dibujo.

- Isaac Lab 2.3 documenta spawners de mallas parametrizadas (cilindro, cono, cubo, esfera), pero no expone `TriangleMeshCfg`/`TrimeshObjectCfg` en el API publicado; la referencia de Isaac Lab 3.0 si los incluye y describe casco convexo para mallas dinamicas. Antes de implementar prismas, confirmar la version exacta instalada con Isaac Sim 5.1 y elegir un spawner USD compatible; no usar un cubo/cilindro visual como sustituto de colision poligonal.
- Fuentes versionadas: [Isaac Lab 2.3 spawners](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sim.spawners.html) y [Isaac Lab 3.0 spawners](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/api/lab/isaaclab.sim.spawners.html).
- **Archivos:** `scene_spec.py`, `scene_config.py`, `real_world.py`, spawner en `11_real_sorting_with_monitor.py`, `perceive.py`, `ground.py`, `planner.py`, vocabulario, frontend.
- **Esfuerzo:** cilindro completo 1–2 semanas; prismas 2–3 semanas.
- **Aceptación:** caída/rodadura/contacto caracterizados; detección ≥95% y pick-place ≥90% en 30 trials antes de habilitar shape; estrategia no validada aparece deshabilitada.

### P3 — World model

No entrenar ahora. Reconsiderar solo si P0/P2 encuentra fallos repetibles que reglas/geometry no describen; primero recoger dataset y demostrar valor medible con el gate de sección 6.

### P4 — Pulido frontend

**Aplicado:** modal responsivo, escena visual editable, pausa, biblioteca, camara con antiguedad, plan, causas de agarre, reset/undo y validacion visible. **Pendiente:** modo de presentacion y validacion en viewport movil/teclado; aun falta verificacion manual de 1366x768 y 390 px.

- **Archivos:** monitor HTML, servidor, `frame_encoder.py`, `telemetry.py`.
- **Esfuerzo:** 2–3 días. Riesgo: duplicar estado; el frame antiguo no debe parecer percepción actual.
- **Aceptación:** usable 1366×768 y 390 px; pausar/reanudar/stop y guardar/cargar se entienden; ninguna mutación de escena mientras el robot se mueve.

## 9. Evaluación repetible

Fixtures/semillas: dos cubos separados; cubo+bola; mismo color; poses cercanas; oclusión parcial por dedos; bola al límite de agarre; cada geometría nueva. Aleatorizar dentro de envolvente realmente alcanzable, fijar configuración/semilla y guardar verdad, cámara, frame, resultado y fallos. Al menos 20 semillas para detección y 30 intentos por geometría para manipulación.

| Métrica | Definición |
|---|---|
| Precisión/recall de instancia | Emparejamiento por IoU y clase/color; separada por forma/luz/oclusiones. |
| Error XY | P50/P95/máximo en mm contra pose de verdad; centro y huella separados. |
| Éxito de agarre | Levantamientos confirmados/intentados; registrar movimiento durante cierre y lift. |
| Éxito de tarea | Objetivos verificados en recipiente correcto/solicitados; no inferirlo del detector. |
| Recuperación | Fallos transitorios corregidos/fallos detectados; reintentos por entidad y causa. |
| Latencia | P50/P95 captura→detección, detección→link, comando→movimiento, duración por objeto y antigüedad de frame. |

No mezclar ground truth de Isaac con la salida visual. Reportar abortos, timeout y resultados parciales; guardar fotogramas fallidos. Secuencia: benchmark offline → cámara quieta → pick/place → escena completa.

## 10. Compatibilidad y fuentes

- [Isaac Lab: cámaras/sensores](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html): RGB, depth y pinhole.
- [Isaac Lab: spawners](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.spawners.html): cubo, esfera, cilindro; no confirma estrategia de agarre SO-101.
- [LeIsaac](https://github.com/LightwheelAI/leisaac) y [documentación LeRobot EnvHub/LeIsaac](https://github.com/huggingface/lerobot/blob/main/docs/source/envhub_leisaac.mdx): integración SO-101/IsaacLab y flujos de datos.
- [Ultralytics segmentación de instancias](https://docs.ultralytics.com/tasks/segment): máscaras/clase/confianza; usar solo si el benchmark justifica dependencia.
- [NVIDIA Isaac Sim/Replicator](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_sim_reference_architecture.html): anotaciones de cajas, instancia, depth y randomización; sintéticos son entrenamiento, no evaluación real.

## Recomendación final

Cerrar P0 con fixtures de poses aleatorias y campaña específica de agarre; completar P1 con validación visible, tamaño de recipientes, reset y preflight; habilitar cilindro únicamente tras validar spawn→visión→grasp→placement; dejar prismas detrás de pruebas físicas y funcionales. No instalar YOLO ni entrenar JEPA para el caso inmediato. Terminar P4 con monitor de percepción/plan español y verificación etiquetada como verdad de simulación.

Arquitectura recomendada: **editor cenital 2D + catálogo explícito + OpenCV/depth + asociación/seguimiento con incertidumbre + planificación IK determinista + verificación física y recuperación limitada**. Es la opción más simple y medible compatible con las restricciones del hackathon.

## Estado de continuación

En la continuación de implementación se añadieron:

- `POST /api/config/validate`, que construye el mismo catálogo que usa Isaac sin guardar ni iniciar sesión. El editor lo llama tras los cambios, muestra errores de alcance/solape y descarta respuestas obsoletas.
- Edición de dimensiones exteriores X/Y/altura, pared y fondo de recipientes; la cavidad se recalcula para mantener pared uniforme. Añadidos deshacer/rehacer, reset al preset demo e importar/exportar JSON.
- La tabla visual muestra el score de calidad de blob que ya calcula `perceive.confidence_of`; no se presenta como una probabilidad calibrada.
- Al rechazar un agarre o perder la pieza en el ascenso, el worker publica una causa en español (apertura geométrica, centrado, contacto/retención) y medidas disponibles de ese intento; el panel la muestra junto a cámara.

El editor ya expone una geometria cilindrica experimental: el validador, el spawner nativo de Isaac, el catalogo, la asociacion de percepcion y las reglas de orden por tipo/color tienen soporte inicial. No se incluye en el preset de demo ni se considera validada: falta ejecutar en Isaac Sim spawn, contacto, cierre, retencion, transporte y deposito en al menos 30 intentos. La interfaz avisa que el agarre no esta validado; los prismas siguen pendientes. Tambien faltan el dataset repetible de trials y la medicion/uso de antiguedad de frame dentro del servo.

- Cuando la validacion rechaza una pose, el mapa resalta en rojo las entidades identificables por el error del backend; al editar y revalidar, el resaltado se limpia.
- Durante `loading`, `starting`, ejecucion y parada, el editor de escena queda inerte y sus acciones de guardar/iniciar se deshabilitan. Las acciones calientes permanecen disponibles; al detenerse, la edicion vuelve a habilitarse.

La compatibilidad de comandos explicitos para cilindros se limita a las reglas completas del parser (por ejemplo, tipo y color); el fallback del modelo de lenguaje no se considera entrenado para esa geometria. Si la orden no se reconoce de forma determinista, debe rechazarse/reformularse, no ejecutar una inferencia ambigua.

En la siguiente continuación de percepción se añadieron:

- El split de color consulta el área proyectada máxima declarada en la escena además del ancho de la mancha; los grupos que exceden el tamaño de una pieza y no se pueden separar con suficiente evidencia quedan señalados como ambiguos en vez de hacerse pasar por un objeto único.
- Las detecciones ambiguas no pueden emparejarse con el catálogo ni satisfacer una descripción de grounding. Si la cámara primaria encuentra una, se intenta una observación desde la cámara de respaldo; si sigue sin resolverse, el panel lo informa como “por reobservar”.
- Las observaciones de escena guardan `captured_at`; `locate()` rechaza una observación de más de un segundo antes de ejecutar un agarre. Esto evita usar una captura caducada como pose actual.
- `tools/benchmark_demo11.py perception` evalúa los umbrales y tamaños de producción, separa las abstenciones y reporta precisión, recall, latencia P95 y error XY P95. Ya se ejecutó con las fixtures versionadas; falta una campaña mayor de capturas aleatorias en Isaac. Los números históricos de otras campañas no se reemplazaron.

### Verificación offline de percepción — 2026-10-18

Se ejecutó `python tools/benchmark_demo11.py perception` con las dos capturas RGB-D versionadas (`container_fragment_top` y `small_triangle_top`), a brillo original y reducido: 4 combinaciones muestra/iluminación, 10 instancias de verdad en total. OpenCV robusto obtuvo recall 0,80, precisión 0,80, 1,68 mm de error XY medio y 9,74 ms de latencia media; dejó 2 regiones como ambiguas. El baseline obtuvo recall 0,60, precisión 0,75, 3,49 mm y 8,36 ms. Esto favorece el modo robusto en este conjunto pequeño, pero no demuestra rendimiento general ni sustituye la evaluación aleatoria con Isaac.

La ejecución descubrió y corrigió un fallo del propio benchmark: sin capturas en `out11/benchmarks/captures`, antes generaba métricas vacías (`truth=0`) sin avisar. Ahora usa las fixtures incluidas y falla con una instrucción explícita si no hay muestras evaluables. Se corrigió además una variable local que ocultaba el módulo `colors` y solo fallaba al evaluar una escena real.

### Verificación de continuación — 2026-10-18

- La campaña física estricta superior de 30 mm completó cinco ejecuciones: bola, cilindro, prisma triangular y hexagonal pasaron en el primer lote; el cubo se detuvo antes del plan por una abstención falsa. Tras ampliar el margen proyectado de área de una huella (factor 0,90 a 1,10, manteniendo el disparador de grupo en 1,30), la captura medida de ese cubo quedó confirmada y el ensayo físico estricto del cubo pasó: **5/5 geometrías trasladadas y verificadas, una repetición cada una, sin agarre asistido**. Es una validación puntual, no una tasa estadística.
- Con 14 fotogramas iniciales RGB-D y 30 instancias de verdad re-evaluadas, OpenCV robusto obtuvo recall 1,00, precisión 0,9375, 28/30 clases correctas, error XY medio 2,27 mm y P95 4,32 mm. La latencia medida después de evitar clustering de profundidad en bandejas cuyo color no aparece entre las piezas configuradas fue 35,0 ms media / 48,6 ms P95. La captura mide CPU y carga de Isaac concurrentes; repetir con máquina ociosa antes de fijar un presupuesto de latencia.
- La campaña reveló que el modo de reset escribía velocidades PhysX sobre el cubo del USD y otros cuerpos sin comprobar si eran cinemáticos. Se añadió lectura de `UsdPhysics.RigidBodyAPI` y el reset ahora escribe velocidades solo en cuerpos confirmados como dinámicos; los cuerpos cinemáticos o no verificables se omiten. Regresión offline añadida; falta confirmar en el siguiente arranque que desaparecen los mensajes nativos de PhysX.
- El arranque inicialmente falló por permiso al escribir el `.kit` bajo el checkout compartido de IsaacLab. El cargador ahora almacena la experiencia temporal en `out11/kit_cache` y conserva la ruta absoluta del directorio `source` para la resolución de extensiones. La campaña de cinco geometrías arrancó tras este cambio y se completó; los fallos de permiso de esa ruta quedan resueltos.

La evaluación actual cubre una pose por geometría en la región delantera, vistas y tamaños limitados. Antes de llamar al agarre generalizado se requieren múltiples semillas, clusters, variantes de cámara, tamaños de 20/30 mm y al menos 30 repeticiones físicas por geometría.


### Continuaci?n de reset y escenas agrupadas ? 2026-10-18

- Se sustituy? el evento `reset_all` de IsaacLab por un reset equivalente para poses, articulaciones y deformables, que omite escrituras de velocidad si el rigid body es cinem?tico o su estado no se puede verificar. La prueba unitaria cubre los tres casos. El primer arranque f?sico posterior mostr? la bandeja `box_red` como cinem?tica y ya no emiti? los errores `setLinearVelocity/setAngularVelocity: Body must be non-kinematic`.
- La escena de cubo agrupado obtuvo detecci?n inicial 4/4, 0 falsos positivos, clases correctas 4/4 y error XY medio/m?ximo 1,52/4,22 mm. La tarea f?sica estricta fall? al centrar la pinza: hubo correcciones de herramienta de hasta 163 mm y se cancel? el cierre con una separaci?n residual de 3,2?3,3 mm, superior al margen. Esto separa un buen resultado de percepci?n de un problema de alcance/servo; no cuenta como ?xito de agarre.
- La repetici?n ampliada de clusters con cinco geometr?as sigue incompleta. No declarar tasa de ?xito para objetos juntos hasta completar prismas y varias semillas en zona central; la escena delantera es sensible a calibraci?n y alcance. Mantener el reporte parcial y conservar frames/logs de fallos.


- Como seguimiento del ensayo, el descenso ahora compara la posici?n observada con el ancla fresca previa a la aproximaci?n; deriva acumulada mayor de 10 mm provoca retirada y no seguimiento de la pieza. La prueba unitaria verifica l?mites durante descenso y movimiento por encima.
- Se retir? del generador la pose central anterior (`x=.348, y=-.400`), porque pas? el chequeo radial pero fall? el servo en simulaci?n. El fixture central usa ahora una pose de recogida ya medida (`x=.380, y=-.400`), y los distractores de color compartido se colocan juntos pero dentro del alcance ensayado. La matriz previa con `seed=11900` fue generada con el fixture anterior y permanece parcial; sus filas no validan la nueva matriz.
- Verificaci?n offline de continuaci?n: 139 checks de matriz de escenas, vocabulario de ?rdenes, asociaci?n, reset de cuerpos cinem?ticos y deriva visual; `py_compile` de los m?dulos modificados. La repetici?n f?sica estricta para el fixture actualizado queda pendiente.


### Campa?a agrupada exploratoria ? resultados parciales

La matriz `continuation_strict_cluster_central_all_shapes_20261018.json` usa el fixture central anterior a la revisi?n de pose. Sus tres primeras filas (cubo, bola y cilindro) tienen detecci?n inicial 4/4, cero falsos positivos y error XY medio aproximado de 1,5 mm, pero las tres tareas estrictas fallaron: el cubo se desplaz? al cerrar/elevar; la bola sigui? 75/90 mm y el transporte se detuvo con 55,5 mm de error de pinza; el cilindro cancel? el cierre a 4,5 mm del centro. No hubo agarres asistidos. El ensayo triangular sigue en ejecuci?n y la fila hexagonal a?n no consta, as? que la campa?a no est? completa y no se publicar? como benchmark final.


La actualizaci?n del fixture central qued? validada offline en 120 combinaciones (3 c?maras ? 2 variantes ? 5 geometr?as ? 2 layouts ? tama?os de 20/30 mm), con todas las escenas construidas y validadas. Tambi?n pasaron 8 regresiones espec?ficas de detecci?n agrupada/solapada, los tests del editor de entorno y su validaci?n, el replay JavaScript del panel, reglas de color del destino y el guard del reset. Esta matriz offline comprueba validez de escenas, no ?xito de agarre; la nueva pose a?n necesita evaluaci?n f?sica.


El cierre ahora tambi?n compara el centro f?sico del objeto despu?s de cerrar con el centro de escena medido justo antes. Si el cierre lo desplaza m?s de 8 mm, marca agarre fallido, cancela la elevaci?n y vuelve al flujo de reobservaci?n. Este guard, el l?mite de deriva de 10 mm durante descenso y el l?mite acumulado de 25 mm durante el movimiento por encima tienen pruebas offline; falta medir la tasa de recuperaci?n en Isaac con el fixture central corregido.

### Ensayo agrupado con bola de 30 mm y almohadilla V experimental — 2026-10-18

Se validó offline una escena central nueva con una bola azul objetivo, dos bolas verdes próximas y una caja roja (`seed=12020`). Se ejecutó en Isaac Sim 5.1 con CPU, agarre estrictamente físico, rigidez 700 y perfil V35 experimental forzado; la asistencia cinemática estuvo desactivada. La detección inicial identificó las tres bolas por separado y la caja. El robot no alcanzó un centrado físico estable: la fase `above` necesitó 12 correcciones del centro de herramienta, con residual máximo reportado de 155 mm; `down` necesitó 11 y reportó hasta 69 mm. El cierre se rechazó al observar el objeto a 616 mm de la posición de pinza; no hubo agarre ni traslado. Por tanto, esta ejecución no evalúa la retención de las almohadillas V: falló antes del contacto útil. El reporte de ejecución quedó en `out11/benchmarks/runtime_12020.json`.

El mismo tamaño de bola había completado un agarre, traslado y depósito físico en una escena aislada (`seed=12005`). La brecha medida está en el flujo de aproximación de la escena agrupada, y requiere revisar el recorrido/articulación y la asociación visual durante el movimiento antes de ajustar fuerza o geometría de las almohadillas. Se mantiene la separación entre resultado físico y asistido. El archivo de matriz que acepta las escenas es una validación geométrica; no convierte este resultado en una tasa de éxito.

Las regresiones locales de esta continuación pasaron para el hover alto de bolas grandes, la separación entre alineamiento lateral y descenso, y la validación reproducible de la escena. Falta reproducir el fallo con traza del centro de herramienta y probar una disposición agrupada alternativa en varias semillas; no declarar resuelto el agarre de bolas agrupadas.

### Aproximación de bola agrupada: servo de hover, oclusión y cierre gradual — 2026-10-18

Se añadió un centrado amortiguado adicional con la pinza abierta, a altura de hover y antes del descenso. Permite hasta 24 correcciones de 8 mm, exige quedar dentro de 8 mm y aborta el descenso si no converge. La ausencia de detección en la franja de oclusión tardía conserva el último ancla durante un máximo de dos observaciones. Se agregó un margen de captura de 4,5 mm exclusivamente como override experimental para bola de 30 mm con almohadillas V; el margen medido de producción sigue siendo 3,1 mm. También se añadió una rampa de cierre de 45 pasos para bola de 30 mm, optativa por variable de entorno.

La repetición física estricta `seed=12029`, escena agrupada superior, bola azul de 30 mm, V35, rigidez 700, rampa 45 y margen experimental 4,5 mm completó **1/1 depósito verificado, un agarre físico y cero asistidos**. La bola siguió el 97 % de la elevación (87/90 mm); el error físico del objeto durante traslado fue 5,8 mm. En 90 pasos de cierre, el centro del objeto permaneció dentro de unos 5,6 mm del centro de apertura. Evidencia: `out11/benchmarks/runtime_12029.json` y `results/continuation_cluster_ball_vee35_slack45_ramp45_lateocclusion60_12029.log`.

No es aún una solución robusta: la repetición `seed=12030` abortó durante la aproximación al perder una observación antes del contacto; `seed=12028` también abortó en descenso por una detección ausente. Los intentos previos `12022` y `12023` expusieron residual alto de servo y centrado incompleto; la geometría V no debe activarse automáticamente para bolas de 30 mm hasta completar ensayos repetidos en semillas y disposiciones distintas. El ensayo exitoso demuestra viabilidad del perfil, no una tasa general.

Las regresiones offline pasan para centrado amortiguado y acotado, margen experimental limitado a ball30/V, oclusión tardía de dos observaciones, parada por atasco y rampa de cierre optativa. Pendiente: ejecución estricta de la batería completa, repeticiones físicas adicionales (incluidas cajas/variantes de cámara) y ajuste de la estabilidad del servo ante frames ausentes antes de promover perfiles.

### Repetición del seed fallido y cadencia visual — resultado A/B

La traza del seed 12030 mostró que la bola se desplazó 36,4 mm durante el descenso, antes de cerrar; no era un error de clasificación de identidad en la captura inicial. Se probó aumentar la frecuencia de observación de 10 a 5 pasos para bolas de 30 mm. Esa variante empeoró la estabilidad del servo: el centrado se detuvo en la fase `above` después de 40 correcciones por pose inalcanzable. El cambio de cadencia se revirtió; Demo 11 conserva 10 pasos.

Estado físico del fixture superior agrupado (bola azul 30 mm, V35/rampa/margen experimentales): un éxito físico verificado (`12029`), al menos tres abortos anteriores a contacto útil (`12022`, `12023`, `12028`) y el seed `12030` abortado tras desplazar la bola en el descenso; una repetición de `12030` con cadencia 5 se detuvo aún antes en `above`. El perfil no está listo para activarse automáticamente. Próximo trabajo: reducir el desvío XY dependiente de Z con calibración/modelo cinemático de la apertura y trayectoria de pre-contacto, manteniendo la retirada si el objeto se mueve.

### Continuación: límite de velocidad vertical tras espera de precontacto — 2026-10-18

La traza detallada del intento `seed=12030` aclaró una causa adicional y reproducible: el plan nominal seguía avanzando mientras la orden de Z permanecía pausada a 0,285 m. Al liberar la espera con el error lateral dentro de 8 mm, el comando podía saltar al Z acumulado del plan; la siguiente observación encontró un cambio de apertura de aproximadamente 72 mm. La corrección en `11_real_sorting_with_monitor.py` limita ahora el avance vertical descendente a 1 mm por paso, solo para bolas de 30 mm, incluso después de liberar la pausa. No cambia las otras geometrías ni la trayectoria normal cuando los pasos ya son menores.

Compilación y 15 regresiones enfocadas pasaron, incluida una que comprueba el caso de liberación con 62 mm acumulados. La batería detectó además un doble de prueba sin campo `kind`; el acceso se hizo tolerante y la batería volvió a pasar. Aún no se ha ejecutado una nueva prueba física con esta corrección; hasta medirla en Isaac, el resultado es una mejora verificada por unidad y una hipótesis física pendiente, no una solución del agarre agrupado. Se mantienen la retirada segura ante deriva y los perfiles V/margen/rampa experimentales.

### Arranque desde el Python base de LeRobot: dependencia opcional ausente — 2026-10-18

Los tres logs de sesión más recientes (`out11/sessions/7bece5bbb14e4f0c85bc81599605a2c6`, `6f54267cadff42fe909e346474f87b3f` y `d23c7c544dc045d8a854163b4bc6b855`) contienen el mismo traceback antes de crear el entorno: `CommandModel.__init__` falla con `ModuleNotFoundError: No module named 'torch'`. El `python` base activo es `C:\Users\Usuario\AppData\Local\Programs\Python\Python312\python.exe` y no tiene PyTorch. Isaac Sim no fue lanzado; el mensaje de tres intentos atribuía por error el fallo Python al simulador.

En modo `hybrid` ahora se continúa con el parser determinista si falta únicamente PyTorch. Ese parser acepta comandos explícitos completos y se abstiene ante órdenes ambiguas; el panel registra el fallback. El backend `model` explícito conserva un error directo si falta la dependencia. Además, el supervisor termina sin reintentos cuando el worker publica una excepción Python antes de que Isaac esté listo; los fallos nativos de Kit siguen usando el reintento acotado. Verificación: compilación y pruebas enfocadas de fallback, extracción de color/tipo/destino y clasificación de error pasaron. Falta que el usuario reintente desde el panel para confirmar el arranque real de Isaac con el entorno instalado.

La batería `tests/test_demo11.py` se intentó completa. Alcanzó el informe de 1338 checks y 12 fallos mientras la lógica de hover aún se aplicaba ampliamente; después se restringió al perfil de bola de 30 mm y se volvieron a ejecutar aisladamente 13 pruebas relevantes, todas pasan. El proceso integral no terminó: Isaac Kit notificó `STATUS_ENTRYPOINT_NOT_FOUND` y procesos hijos de sesión falsa fallaron al crear logs temporales. No se interrumpieron procesos Python/Kit.
