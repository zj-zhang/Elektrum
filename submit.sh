#!/bin/bash

while getopts 'abcde:h' opt; do
  case "$opt" in
    a)
	echo "Submit simulated runs"
	# 1. Amber Kinn on simulated data
	for i in `seq 1 3`; do
		#sbatch -J ka_ka$i sbatch_AmberSimKinn.sh 21-11-1_test_1 rep$i #sbatch -J ka_noka$i sbatch_AmberSimKinn.sh 21-11-1_test_1 rep$i --disable-posterior
		#sbatch -J del_ka$i sbatch_AmberSimKinn.sh 22-06-28_synth_data_depl rep$i
		#sbatch -J del_noka$i sbatch_AmberSimKinn.sh 22-06-28_synth_data_depl rep$i --disable-posterior
		sbatch -J del_eig$i sbatch_AmberSimKinn.sh 22-06-28_synth_data_depl rep$i --use-sink-state
		sbatch -J del_noeig$i sbatch_AmberSimKinn.sh 22-06-28_synth_data_depl rep$i --use-sink-state --disable-posterior
	done
	;;
    b)
	echo "Submit Cas9 w/ Finkelstein model space runs"
	# 2. Amber Kinn - Finkelstein model space
	for i in `seq 1 5`; do
		sbatch -J wt-f-0$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log finkelstein 0 0 rep$i-gRNA1
		sbatch -J wt-f-1$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log finkelstein 0 1 rep$i-gRNA2
		#sbatch -J wt-f-0$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log finkelstein 0 0 rep$i-gRNA1 --use-sink-state
		#sbatch -J wt-f-1$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log finkelstein 0 1 rep$i-gRNA2 --use-sink-state
	done
	;;
    c)
	echo "Submit Cas9 w/ Uniform model space runs"
	# 3. Amber Kinn - Uniform anchoring
	for i in `seq 1 3`; do
		for ns in `seq 4 6`; do
			sbatch -J wt-$ns-0$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log uniform $ns 0 rep$i-gRNA1
			sbatch -J wt-$ns-1$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log uniform $ns 1 rep$i-gRNA2
		done
	done
	;;
    d)
	echo "Submit Cas9 CNN"
	# 4. Amber Cnn
	for i in `seq 1 5`; do
		sbatch -J wt0_$i sbatch_AmberCnn.sh wtCas9_cleave_rate_log 0 rep$i-gRNA1
		sbatch -J wt1_$i sbatch_AmberCnn.sh wtCas9_cleave_rate_log 1 rep$i-gRNA2
	done
	;;
    e)
	echo "Submit Transfer Learning CNN+KINN"
	# 5. Amber Transfer Learning CNN+KINN
	for i in `seq 1 3`; do
		sbatch -J TL$i sbatch_AmberTL.sh TL_$i
	done
	;;
    ?|h)
        echo "Usage: $(basename $0) [-a] [-b] [-c] [-d]"
        exit 1
        ;;
  esac
done



