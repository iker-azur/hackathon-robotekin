#!/bin/bash
cd "C:/Users/Usuario/Documents/Workspace/LeRobot_/demos/05_planner_and_executor"
PY="C:/Users/Usuario/miniconda3/envs/hackathon-robotekin/python.exe"
export HF_HUB_OFFLINE=1
for s in 32000 33000; do
  $PY tools/benchmark_scene_matrix.py --headless --strict-physical --cameras top front side --variants open contained --geometries triangular_prism --layouts single cluster --sizes-mm 20 --repetitions 1 --region central --seed $s --output results/tri20_val_$s.json > results/tri20_val_$s.out 2>&1
done
echo done > results/tri20_val.done
