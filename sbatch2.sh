#!/bin/bash
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --partition=gpu
##SBATCH --partition=genx
#SBATCH -c 4 --mem=64gb
#SBATCH --time 2-0:00:00

export PYTHONPATH="."
target=$1
id=$2
switch=$3
mkdir -p outputs/CNN-$target-$id
python notebooks/runAmber_cnn.py --target $target --wd outputs/CNN-$target-$id --switch $switch > outputs/CNN-$target-$id/log.txt
