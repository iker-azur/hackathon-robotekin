#!/bin/bash
# 20 mm triangle variant sweep on the 12-scene baseline set (seed 31000)
cd "C:/Users/Usuario/Documents/Workspace/LeRobot_/demos/05_planner_and_executor"
PY="C:/Users/Usuario/miniconda3/envs/hackathon-robotekin/python.exe"
COMMON="--headless --strict-physical --cameras top --variants open --geometries triangular_prism --layouts single cluster --sizes-mm 20 --repetitions 6 --region central --seed 31000"
export HF_HUB_OFFLINE=1
DEMO11_TRIANGLE_CLOSE_RAMP_STEPS=45 $PY tools/benchmark_scene_matrix.py $COMMON --output results/tri20_ramp45.json > results/tri20_ramp45.out 2>&1
$PY tools/benchmark_scene_matrix.py $COMMON --close-command 0.06 --output results/tri20_close006.json > results/tri20_close006.out 2>&1
DEMO11_TRIANGLE_CLOSE_RAMP_STEPS=45 $PY tools/benchmark_scene_matrix.py $COMMON --close-command 0.07 --output results/tri20_ramp45_close007.json > results/tri20_ramp45_close007.out 2>&1
echo done > results/tri20_variants.done
