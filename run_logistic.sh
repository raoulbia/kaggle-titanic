#!/bin/bash -x

python3 src/main.py \
    --dataset titanic \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/titanic-train-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/titanic-test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/titanic-results.csv \
    --test-size 0.3 \
    --num-iters 1000 \
    --cost-history-plot -1 \
    --learning-curve 1 \
    --learning-rate 0.3 \
    --reg-param 1.1 \
