# Resumen de evaluaciones

Generado por `tools/23_summarize_results.py` a partir de `results/runs/*.json`; no se edita a mano. Cada ejecución del demo 16 deja su propio archivo y nunca sobrescribe otro.

| fecha (UTC) | ejecución | tipo | entradas | cubos | éxitos | fallos |
|---|---|---|---|---|---|---|
| 2026-10-19T16:07:24Z | [`scripted_bias_heldout_10ep_try1`](runs/scripted_bias_heldout_10ep_try1.md) | scripted | SCRIPTED teacher with the cube's true pose (privileged) - NOT a learn… | heldout | **10/10** | - |
| 2026-10-19T17:39:19Z | [`act_clock_val_40demos_heldout_10ep`](runs/act_clock_val_40demos_heldout_10ep.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **10/10** | - |
| 2026-10-19T17:59:04Z | [`act_clock_final_heldout_10ep`](runs/act_clock_final_heldout_10ep.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **10/10** | - |
| 2026-10-19T18:35:34Z | [`act_clock_final_heldout_from10_20ep`](runs/act_clock_final_heldout_from10_20ep.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **20/20** | - |
| 2026-10-19T18:41:07Z | [`act_clock_final_heldout_10ep_repeat`](runs/act_clock_final_heldout_10ep_repeat.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **10/10** | - |
| 2026-10-19T18:45:45Z | [`scripted_original_grasp_point_heldout_10ep`](runs/scripted_original_grasp_point_heldout_10ep.md) | scripted | SCRIPTED teacher with the cube's true pose (privileged) - NOT a learn… | heldout | **10/10** | - |
| 2026-10-19T18:48:39Z | [`baseline_act_phase_heldout_5ep`](runs/baseline_act_phase_heldout_5ep.md) | act | camera image + joint state + PRIVILEGED plan phase index | heldout | **0/5** | misplaced×1, never_lifted×4 |
| 2026-10-19T18:52:57Z | [`baseline_act_cube_detect_heldout_5ep`](runs/baseline_act_cube_detect_heldout_5ep.md) | act | camera image + joint state + cube pixel centroid from the colour dete… | heldout | **0/5** | never_lifted×5 |
| 2026-10-19T18:55:47Z | [`baseline_act_image_state_heldout_5ep`](runs/baseline_act_image_state_heldout_5ep.md) | act | camera image + joint state (nothing privileged) | heldout | **0/5** | never_lifted×5 |
| 2026-10-20T15:32:35Z | [`act_clock_zip_roundtrip_heldout_2ep`](runs/act_clock_zip_roundtrip_heldout_2ep.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **2/2** | - |
| 2026-10-20T15:37:59Z | [`act_clock_final_wide_10ep`](runs/act_clock_final_wide_10ep.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | wide | **10/10** | - |
| 2026-10-20T15:42:22Z | [`scripted_wide_10ep`](runs/scripted_wide_10ep.md) | scripted | SCRIPTED teacher with the cube's true pose (privileged) - NOT a learn… | wide | **10/10** | - |
| 2026-10-20T15:47:36Z | [`act_clock_final_outside_10ep`](runs/act_clock_final_outside_10ep.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | outside | **8/10** | misplaced×1, never_lifted×1 |
| 2026-10-20T15:52:05Z | [`scripted_outside_10ep`](runs/scripted_outside_10ep.md) | scripted | SCRIPTED teacher with the cube's true pose (privileged) - NOT a learn… | outside | **10/10** | - |
| 2026-10-20T17:16:33Z | [`15_act_clock-last_heldout_1ep_20261020-171521`](runs/15_act_clock-last_heldout_1ep_20261020-171521.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **1/1** | - |
| 2026-10-20T17:18:07Z | [`15_act_clock-last_heldout_1ep_20261020-171649`](runs/15_act_clock-last_heldout_1ep_20261020-171649.md) | act | PRIVILEGED plan clock + cube pixel position seen once at the start by… | heldout | **1/1** | - |

## Comprobaciones sin simulador (`tools/22_validate_offline.py`)

| ejecución | política | episodios | error xy al cerrar la pinza (mm) | error z (mm) |
|---|---|---|---|---|
| `act_clock_val_10k_heldout_40-47` | `15_act_clock_val` | 40-47 | media 3.4, peor 5.0 | media 0.3, peor 0.4 |
