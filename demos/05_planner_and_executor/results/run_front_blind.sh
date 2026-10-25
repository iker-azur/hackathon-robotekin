#!/bin/bash
cd "C:/Users/Usuario/Documents/Workspace/LeRobot_/demos/05_planner_and_executor"
PY="C:/Users/Usuario/miniconda3/envs/hackathon-robotekin/python.exe"
export HF_HUB_OFFLINE=1
A="--headless --strict-physical --cameras top front side --variants open contained --geometries triangular_prism --layouts single cluster --sizes-mm 20 --repetitions 1 --region central"
$PY tools/benchmark_scene_matrix.py $A --seed 33000 --only-seeds 33004 33006 33007 --output results/front_hover_33.json > results/front_hover_33.out 2>&1
$PY tools/benchmark_scene_matrix.py $A --seed 32000 --only-seeds 32005 32006 --output results/front_hover_32.json > results/front_hover_32.out 2>&1
echo done > results/front_hover.done
