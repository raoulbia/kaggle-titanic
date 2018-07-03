#!/bin/bash -x

python3 src/model_train.py \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/train-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/results.csv \
    --test-size 0.4 \
    --num-iters 1000 \
    --learning-rate 0.001 \
    --reg-term 1 \
    --apply-to-kaggle 1