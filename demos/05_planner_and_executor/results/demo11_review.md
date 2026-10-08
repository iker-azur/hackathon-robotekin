# Revisi?n de Demo 11

Fecha: 2026-10-08. Rama: prepare-demos. Entorno: hackathon-robotekin.

## Hallazgos antes de cambios

- Cr?tico: ground.py:370-372 sustituye una caja de color inexistente por otra; reproducido verde ? roja.
- Alto: 11_real_sorting_with_monitor.py:1871 declara ?xito de inicio antes del arranque; :2073 deja la carga del modelo fuera del manejador de errores.
- Alto: :1660-1678 no finaliza ?rdenes de texto.
- Alto: :976,:1268,:1298 reutiliza veredictos de piezas entre ?rdenes.
- Alto: :2242-2248 cierra el panel con Isaac.
- Alto: :1396,:1424,:1463 silencia errores de c?mara/asociaci?n.
- Alto: command_model_v2/evaluate.py:77 omite sort_filter; 48/136 frases de test_real est?n literalmente en train_commands.
- Medio: perceive.py:175,:359 usa operaciones sobre el fotograma completo y umbrales de ?rea; scene_spec.py:318 fija la escena.

## L?nea base medida

- test_demo11.py: 461 comprobaciones, SUCCESS; test_planner.py: 1468 comprobaciones, 20 escenas, SUCCESS.
- Modelo: 68/88 comandos exactos (77,3 %); 81/88 intenciones (92,0 %), media 100,6 ms. Excluidas coincidencias textuales; no comprobada independencia sem?ntica. Evaluador antiguo no incluye sort_filter.
- Isaac: orden bola roja ? caja roja verificada; 1 tarea, 0 sujeciones cinem?ticas. Escena inicial: 4 piezas y 2 recipientes detectados, tipos correctos en 3/4 piezas.
- Perfil detallado: proceso 76,439 s; look media 42,405 ms/15 llamadas, capture_rgb media 21,864 ms/43 llamadas. Incluye carga, arranque y cierre; ejecuci?n headless sin navegador.
- Websocket aislado: 100 publicaciones ? 1 snapshot; primera actualizaci?n 12,595 ms; 0 mensajes extra en 0,5 s inactivo.
- Primer arranque online falla por certificado HTTPS. HF_HUB_OFFLINE=1 permite usar la cach?.
- Navegador real pendiente; las pruebas originales usan Node/DOM simulado y HTTP/websocket.

No hay n?meros despu?s de cambios todav?a.
