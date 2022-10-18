#!/bin/bash
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --partition=gpu
##SBATCH --partition=genx
#SBATCH -c 4 --mem=64gb
#SBATCH --time 2-0:00:00

export PYTHONPATH="."
date=$(date '+%Y-%m-%d')
wd=$1
mkdir -p outputs/$date/$wd
python src/transfer_learn.py --wd outputs/$date/$wd > outputs/$date/$wd/log.txt
