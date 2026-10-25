#!/bin/bash
cd "C:/Users/Usuario/Documents/Workspace/LeRobot_/demos/05_planner_and_executor"
PY="C:/Users/Usuario/miniconda3/envs/hackathon-robotekin/python.exe"
export HF_HUB_OFFLINE=1
A="--headless --strict-physical --cameras top front side --variants open contained --geometries triangular_prism --layouts single cluster --sizes-mm 20 --repetitions 1 --region central"
for d in 0.02 0.035; do
  DEMO11_APPROACH_DROP_M=$d $PY tools/benchmark_scene_matrix.py $A --seed 33000 --only-seeds 33004 33005 33006 33007 --output results/drop_${d}_33.json > /dev/null 2>&1
  DEMO11_APPROACH_DROP_M=$d $PY tools/benchmark_scene_matrix.py $A --seed 32000 --only-seeds 32004 32005 32006 32007 --output results/drop_${d}_32.json > /dev/null 2>&1
done
echo done > results/drop_sweep.done
