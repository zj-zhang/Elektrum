#!/bin/bash
##SBATCH --gres=gpu:v100-32gb:1
##SBATCH --partition=gpu
#SBATCH --partition=genx
#SBATCH -c 4 --mem=64gb
#SBATCH --time 3-0:00:00

export PYTHONPATH="."
target=$1
ms=$2
ns=$3
switch=$4
final_id=$5
use_sink_state=$6
mkdir -p outputs/final/KINN-$target-$ms-$ns-$final_id"$use_sink_state"
python src/runAmber_kinn.py --target $target --ms $ms --n-states $ns --wd outputs/final/KINN-$target-$ms-$ns-$final_id"$use_sink_state" --switch $switch $use_sink_state > outputs/final/KINN-$target-$ms-$ns-$final_id"$use_sink_state"/log.txt
