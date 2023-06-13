#!/bin/bash

if [ $# -eq 0 ]
    then
    env_name="llpsdl"
else
    env_name=$1
fi

conda create -n $env_name
eval "$(conda shell.bash hook)"
conda activate $env_name

conda env update --file dependencies.yaml

pip install torch
pip install torchvision
pip install lightning
