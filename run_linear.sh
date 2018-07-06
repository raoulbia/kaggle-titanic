#!/bin/bash -x
#    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/train-clean.csv \
python3 src/model_train.py \
    --dataset houses \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/data/titanic-train-clean3.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/data/titanic-test-clean3.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/loca-data/titanic-results.csv \
    --test-size 0.3 \
    --num-iters 1000 \
    --cost-history-plot 1 \
    --learning-curve -1 \
    --learning-rate 0.01 \
    --reg-param 1.1 \
