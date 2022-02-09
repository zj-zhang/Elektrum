#!/bin/bash
##SBATCH --gres=gpu:v100-32gb:1
##SBATCH --partition=gpu
#SBATCH --partition=genx
#SBATCH -c 4 --mem=64gb
#SBATCH --time 6-0:00:00

export PYTHONPATH="."
target=$1
ms=$2
ns=$3
switch=$4
mkdir -p outputs/KINN-$target-$ms-$ns
python notebooks/runAmber_kinn.py --target $target --ms $ms --n-states $ns --wd outputs/KINN-$target-$ms-$ns --switch $switch > outputs/KINN-$target-$ms-$ns/log.txt
