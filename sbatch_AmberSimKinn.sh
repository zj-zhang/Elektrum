#!/bin/bash
##SBATCH --gres=gpu:v100-32gb:1
##SBATCH --partition=gpu
#SBATCH --partition=genx
#SBATCH -c 4 --mem=64gb
#SBATCH --time 3-0:00:00

export PYTHONPATH="."
date=$(date '+%Y-%m-%d')
target=$1
fid=$2
use_sink_state=$3
mkdir -p outputs/sim/$date/$target-$fid"$use_sink_state"
python src/runAmber_simkinn.py --max-gen 200 --patience 50 --samps-per-gen 5 \
       	--wd outputs/sim/$date/$target-$fid"$use_sink_state" --data-file ./data/sim_data/$target/data.tsv --param-file ./data/sim_data/$target/params.yaml $use_sink_state #> outputs/sim/$date/$target-$fid"$use_sink_state"/log.txt
