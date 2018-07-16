#!/bin/bash -x

#--alpha 0.3 \
#--_lambda 1.1 \

python3 src/main.py \
    --train-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/titanic-train-clean.csv \
    --test-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/titanic-test-clean.csv \
    --results-file-path /home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/titanic-results.csv \
    --num-iters 1500 \
    --learn-hyperparameters -1 \
    --alpha 0.001 \
    --_lambda 0.03 \
