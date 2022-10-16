#!/bin/bash
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --partition=gpu
##SBATCH --partition=genx
#SBATCH -c 4 --mem=64gb
#SBATCH --time 2-0:00:00

export PYTHONPATH="."
date=$(date '+%Y-%m-%d')
target=$1
fid=$3
switch=$2
mkdir -p outputs/$date/CNN-$target-$fid
python src/runAmber_cnn.py --target $target --wd outputs/$date/CNN-$target-$fid --switch $switch > outputs/$date/CNN-$target-$fid/log.txt
