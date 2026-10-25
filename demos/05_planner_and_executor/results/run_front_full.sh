#!/bin/bash
cd "C:/Users/Usuario/Documents/Workspace/LeRobot_/demos/05_planner_and_executor"
PY="C:/Users/Usuario/miniconda3/envs/hackathon-robotekin/python.exe"
export HF_HUB_OFFLINE=1
A="--headless --strict-physical --cameras front --variants open contained --layouts single cluster --sizes-mm 20 30 --repetitions 1 --region central --seed 34000"
$PY tools/benchmark_scene_matrix.py $A --output results/frontfull_new.json > /dev/null 2>&1
DEMO11_FRONT_PARK_POSE=0.20,-0.58,0.39 $PY tools/benchmark_scene_matrix.py $A --output results/frontfull_old.json > /dev/null 2>&1
echo done > results/frontfull.done
