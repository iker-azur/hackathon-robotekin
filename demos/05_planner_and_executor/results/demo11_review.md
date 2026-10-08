# Revisión inicial de Demo 11

Fecha: 2026-10-08. Rama: `prepare-demos`. Python: `hackathon-robotekin`.
Las referencias siguientes corresponden al código anterior a `979fd5b`; las líneas cambian durante la implementación.

## Hallazgos por gravedad

| Gravedad | Referencia original | Problema observado |
|---|---|---|
| Crítica | `ground.py:370–372` | Un destino de color inexistente se sustituye por otro; reproducido verde → roja. |
| Alta | `11_real_sorting_with_monitor.py:1871,2073` | Inicio marcado correcto antes de arrancar; carga del modelo fuera del manejador de errores. |
| Alta | `11_real_sorting_with_monitor.py:1660–1678` | Las órdenes de texto no alcanzan un estado terminal. |
| Alta | `11_real_sorting_with_monitor.py:976,1268,1298` | Veredictos reutilizados entre órdenes, con riesgo de contabilizar éxito previo. |
| Alta | `11_real_sorting_with_monitor.py:2242–2248` | El panel termina al cerrar Isaac. |
| Alta | `11_real_sorting_with_monitor.py:1396,1424,1463` | Excepciones de cámara/asociación silenciadas. |
| Alta | `command_model_v2/evaluate.py:77` | La comparación omite `sort_filter`; hay frases del conjunto de prueba en entrenamiento. |
| Media | `perceive.py:175,359`; `scene_spec.py:318` | Trabajo por componente sobre toda la imagen; escena y vocabulario fijos. |

## Línea base observada

- Pruebas: Demo 11, 461 comprobaciones, SUCCESS; planner, 1468 comprobaciones sobre 20 escenas, SUCCESS.
- Evaluación antigua: 68/88 comandos exactos y 81/88 intenciones; media 100,584 ms. Excluye 48 coincidencias literales de 136 frases, pero omite `sort_filter`; no comparar directamente con el evaluador corregido.
- Isaac headless: una bola roja colocada y verificada, sin sujeción cinemática. Primera escena: cuatro piezas y dos recipientes; tres de cuatro tipos de pieza correctos.
- Perfil: proceso 76,439 s; `look`, 15 llamadas, media 42,405 ms; `capture_rgb`, 43 llamadas, media 21,864 ms. Incluye carga, arranque y cierre. Entrada a ejecución: 137,145 ms; no mide HTTP hasta el primer movimiento físico.
- Websocket aislado: 100 publicaciones, primer snapshot en 12,595 ms, cero mensajes adicionales durante 0,5 s inactivo.
- Arranque online: fallo de certificado HTTPS. Con `HF_HUB_OFFLINE=1` se pudo usar la caché local.

## Orden de trabajo aprobado

1. Corregir destinos, estados terminales y contabilidad por orden.
2. Mantener el servidor y aislar cada sesión Isaac en un proceso.
3. Añadir catálogo configurable y destinos con nombre.
4. Evaluar percepción, parser y coste de cámara/panel; aplicar mejoras medidas.
5. Verificar HTTP, reinicio y ejecución física; publicar resultados y limitaciones.

Fuera de alcance: modificar Demo 10 o `executor.py`, colores sin calibrar, reentrenar sin evidencia y afirmar rendimiento del navegador sin observarlo.

Resultado y mediciones finales: [demo11_implementation.md](demo11_implementation.md).
