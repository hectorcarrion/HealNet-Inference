#!/bin/sh

USER=$(id -un)
cd /Users/$USER/Desktop/HealNet-Inference
git pull
docker pull hectorcarrion/healnet:0.4
docker run -v /Users/$USER/Desktop:/root/Desktop  hectorcarrion/healnet:0.4
git add prob_table.csv
git commit -m "Table Autocommit"
git push
echo Next HealNet run set in 1hr