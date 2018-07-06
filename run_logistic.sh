#!/bin/bash -x
#    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/data/titanic-train-clean.csv \
python3 src/main.py \
    --model linear \
    --dataset houses-toy \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/house-price-train-toy-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/house-price-test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/house-price-results.csv \
    --test-size 0.3 \
    --num-iters 10 \
    --cost-history-plot 1 \
    --learning-curve -1 \
    --learning-rate 0.1 \
    --reg-param 1 \
