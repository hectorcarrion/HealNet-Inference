#!/bin/sh

USER=$(id -un)
cd /Users/$USER/Desktop/HealNet-Inference
git pull
python3 /Users/$USER/Desktop/HealNet-Inference/docker_run.py
git add prob_table.csv
git commit -m "Table Autocommit"
git push
echo Next HealNet run set in 30m