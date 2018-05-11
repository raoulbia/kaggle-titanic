#!/bin/bash -xe

# this script assumes conda is in $PATH

# the first command line argument specifies the name of the virtual environment
export VIRTUALENV=$1

# if the specified virtual environment exists then remove it
set +e
conda remove -y --name ${VIRTUALENV} --all
set -e

conda create -qy --name ${VIRTUALENV} mkl numpy scipy scikit-learn

source activate ${VIRTUALENV}

#conda update pip

pip install -r requirements.txt

# to remove the virtual environment:
# conda remove -y --name ${VIRTUALENV} --all
