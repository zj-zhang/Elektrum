#!/bin/bash
##SBATCH --gres=gpu:v100-32gb:1
##SBATCH --partition=gpu
#SBATCH --partition=gen
#SBATCH -c 8 --mem=64gb
#SBATCH --time 6-0:00:00

export PYTHONPATH="."
target=$1
ms=$2
ns=$3
mkdir -p outputs/$target-$ms-$ns
python notebooks/03-runAmber.py --target $target --ms $ms --n-states $ns --wd outputs/$target-$ms-$ns > outputs/$target-$ms-$ns/log.txt
