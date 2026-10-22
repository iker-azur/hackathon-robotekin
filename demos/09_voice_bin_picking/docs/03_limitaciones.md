# 3 · Limitaciones que quedan y mejoras recomendadas

Ordenadas por lo que más puede estropear una presentación. «Medido en Isaac» = con `tools/` sobre el simulador físico (22 de octubre de 2026, una GPU RTX 5060 Ti);
«medido sin simulador» = con los mundos de prueba; «sin medir» = depende de un modelo entrenado, de un micrófono real o de un robot real.

## Lo que no está hecho todavía

1. **No hay política ACT entrenada ni YOLO entrenado.** El demo usa el guion del experto, que **no es aprendizaje** (etiqueta fija en el panel y en los informes). La ruta ACT sí se ha
   ejecutado en Isaac, pero con un modelo de 40 pasos que sólo prueba la tubería (0/10 agarres, como se esperaba; ver `04_validacion.md`). Nada se sabe aún de lo que logra una ACT entrenada de verdad.
2. **Faltan las 500 demostraciones.** Medido en Isaac: 3 episodios válidos en 0,7 min (≈ 14 s de reloj por episodio, 100 % de éxito del experto en esa muestra de escenas separadas/mixtas), es decir,
   unas 2 h para 500. La tasa de éxito del experto en pilas y su cobertura de la capa 2 no se conocen con esa muestra.
3. **El agarre por fricción dentro del contenedor es poco fiable.** Medido en Isaac con el guion: ~75 % de los cubos admitidos por el modelo, ~65–70 % de los cilindros, ~37 % de las piezas que el modelo
   del hueco de dedos descarta (el demo 08 midió 30/30 con piezas sueltas sobre la mesa). Los fallos son `deslizada` (se suelta al levantar) y `vacío` (la pinza empuja la pieza al bajar); no
   los causa la percepción (error de posición medio 1,0 mm) ni la holgura de apertura (12, 16 y 20 mm dieron 9, 10 y 10 de 12) ni la profundidad del agarre (cilindros: 7, 8 y 7 de 11 con 0, 3 y 6 mm más abajo).
   Por eso la clasificación completa sólo acierta en torno a **2 de cada 3 escenas** (ver `04_validacion.md`).

## Límites físicos y de diseño (conocidos)

5. **El alcance del SO-101 con la pinza vertical es pequeño** (corona de 0,16–0,23 m). Por eso el contenedor es de 80 × 200 mm y sólo cabe un número pequeño de piezas; la segunda capa
   alcanza el 87 % de sus puntos, no el 100 %.
6. **Con dedos de 8–10 mm, las piezas muy juntas no se pueden coger.** Medido con las escenas del generador (piezas visibles en la primera imagen con hueco para los dedos): 79 % en
   «separadas», 62 % en «mixto», 35 % en «apiladas» y 17 % si *todas* están pegadas en cadena. Una pieza en una esquina, o rodeada de piezas **de otro color**, no se coge nunca porque
   nada le hace sitio: en la lógica de pruebas, «rojas a la izquierda y verdes a la derecha» termina del todo en ~2 de cada 3 escenas y en las demás **lo dice y se detiene** (`incompleto`,
   con el motivo por pieza).
7. **Capacidad de las cajas: 5 piezas cada una** (medido con la cinemática). Una orden que mande más avisa antes de empezar y deja el resto en el contenedor.
8. **Una sola cámara cenital.** Una pieza completamente debajo de otra no existe para ella; una tapada a medias da una posición y un giro menos fiables (`visibility` baja). No se estima la inclinación de
   una pieza ladeada ni se usa la cámara de muñeca (el demo 08 la configura, desactivada).
9. **Máscara del objetivo fija durante el agarre.** Si la pieza se mueve a mitad (un empujón, o la arrastra la pinza al abrir), la política no lo sabe por la máscara, sólo por el RGB.
10. **La recuperación ve fallos al *verificar*, no a mitad de agarre** con ACT; el guion de aproximación del demo 08 vigilaba el movimiento de la pieza durante la aproximación, aquí no se ha portado.

## Percepción y voz

11. **Segmentación HSV medida sobre renders de Isaac** (36 imágenes, 12 escenas; con el brazo plegado, sin brazo delante): precisión 0,92 y recall 0,90 en piezas visibles ≥ 85 % (85 de 94), error de
    posición 1,3–1,6 mm de media, color 100 %, forma 90–93 %. Con el brazo delante baja a recall 0,63 (situación que no ocurre al observar). **Falla en piezas del mismo color que se tocan**
    (dos cubos pegados y escalonados no tienen costura de color ni de profundidad): se reparten por k-means, con un error de posición p95 de 6,5 mm en esas piezas (3,8 mm antes) y una forma peor. En la
    imagen sintética daba 1,0; la diferencia es lo que enseña Isaac. Pierde las visibles < 35 %. No se ha medido sobre cámara real.
12. **Los colores son tres** (rojo, verde, azul): el amarillo se confunde con el brazo. HSV depende de colores saturados y de la iluminación del render.
13. **Las etiquetas de YOLO son analíticas** (siluetas proyectadas). Alinean con el render si la calibración es buena; `export_yolo_dataset.py --debug` guarda contornos para comprobarlo a ojo.
    Una red ajustada así no vale para una cámara real sin volver a ajustarla.
14. **El analizador de órdenes es de reglas.** Frases fuera del vocabulario se rechazan con un mensaje, no se interpretan «de forma flexible». Izquierda/derecha son las del robot, y «Recoge todas las
    azules» sin caja usa la derecha (avisando).
15. **La voz depende del micrófono y del modelo Whisper *small* en CPU.** Medido con la voz sintética de Windows (Microsoft Helena, español): 6 de 6 órdenes entendidas igual que el texto, ~1,3 s por frase.
    No se ha medido con una persona, con acento ni con ruido de sala, ni la grabación real con micrófono (sólo que se detecta). El panel funciona igual con texto.
16. **Una pieza totalmente tapada no existe para la cámara.** Medido en Isaac: en 3 de 24 escenas el robot dijo «nada que hacer» y en 1 «completado» con piezas del color pedido aún debajo de otras. El mensaje ya
    lo advierte («una pieza totalmente tapada por otras no se puede ver»), pero la orden queda sin cumplir hasta que exista la habilidad «apartar» (ver mejoras).

## Datos y política

17. **El criterio de éxito del experto filtra los casos difíciles**: sólo entran episodios en los que la pieza se levantó. La política no verá nunca agarres que el experto no logra, así que su cobertura
    de pilas difíciles es la del experto.
18. **El ruido de aproximación (σ = 0,02 rad) es una elección**, no calibrada. Podría ser poco o demasiado.
19. **ACT infiere en CPU** (cargar CUDA antes de Isaac rompe PhysX en esta máquina); el bucle a 30 Hz con el modelo diminuto corrió en Isaac sin errores, pero la latencia con un ACT entrenado de verdad no está medida.
20. **Una sola variedad de pieza**: cubos y cilindros de 25 mm, 12 g, mismos materiales. El conjunto de piezas es fijo (3 colores × 3 cubos + 1 cilindro: como máximo 4 por color) porque Isaac no añade cuerpos
    con la escena en marcha.
21. **La física es la del demo 08**: rigidez de servos, fuerza de pinza, fricción y masas son suposiciones razonadas, no medidas en un SO-101 real. No hay transferencia a robot real probada.

## Mejoras recomendadas (en este orden)

1. **Mejorar el agarre por fricción dentro del contenedor** (limitación 3): es el cuello de botella medido, por encima de percepción y selección. Sin cambiar la física del demo 08 no hay palanca simple
   (probadas y descartadas: holgura de apertura, profundidad del agarre). Candidatas: dedos con más fricción o superficie, más fuerza de cierre en cilindros, y una recogida más lenta al levantar.
2. **Habilidad «apartar»**: cuando las piezas de la orden están bloqueadas por vecinas o por la pared, o tapadas (limitación 16), coger una vecina agarrable y dejarla en un hueco libre *del contenedor*. Subiría
   mucho la tasa de «completado» (los casos incompletos medidos en Isaac son sobre todo «los dedos no caben» y piezas tapadas). `place.py` ya sabe elegir huecos; habría que generalizarlo al contenedor.
3. **DAgger**: ejecutar la política en el simulador y dejar que el experto etiquete los estados que visita; es la forma estándar de cerrar el hueco entre las demostraciones y el lazo cerrado.
4. **Cámara de muñeca** para el último centímetro del agarre, y **estimación de inclinación** con la profundidad (plano ajustado a la cara superior).
5. **Seguimiento de la máscara en vivo** y política con la máscara actualizada.
6. **Más variedad**: tamaños, formas y colores por pieza (recolorear el material USD en cada reinicio), iluminación aleatoria, ruido de cámara — y con ello sí tendría sentido ajustar YOLO.
7. **Clasificación por defectos**: clasificador en `attributes.py` y piezas con defecto simulado (ver `02_arquitectura.md`).
8. **Demostraciones humanas por teleoperación del SO-101** (más variadas que el experto y la única vía a la imagen real).
9. **Medir la fricción y los servos reales** y recalibrar `recoverykit/config.py`; escribir un `RealRobot` para el `RobotPort`.
10. **Mejor entrada de voz**: detección de actividad más robusta, palabra de activación, y una respuesta hablada.
