#!/bin/sh

USER=$(id -un)
cd /Users/$USER/Desktop/HealNet-Inference
git pull
python3 /Users/$USER/Desktop/HealNet-Inference/healnet_inference.py
git add prob_table.csv
git add log.txt
git commit -m "Table Autocommit"
git push
echo Next HealNet run set in 30m