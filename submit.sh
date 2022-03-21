# Amber Kinn - Finkelstein model space
for i in `seq 1 3`; do
	sbatch -J wt-f-0$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log finkelstein 0 0 rep$i-gRNA1
	sbatch -J wt-f-1$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log finkelstein 0 1 rep$i-gRNA2
done

# Amber Kinn - Uniform anchoring
for i in `seq 1 3`; do
	for ns in `seq 2 5`; do
		sbatch -J wt-$ns-0$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log uniform $ns 0 rep$i-gRNA1
		sbatch -J wt-$ns-1$i sbatch_AmberKinn.sh wtCas9_cleave_rate_log uniform $ns 1 rep$i-gRNA2
	done
done

# Amber Cnn
#sbatch -J wt0 sbatch_AmberCnn.sh wtCas9_cleave_rate_log 0 0
#sbatch -J wt1 sbatch_AmberCnn.sh wtCas9_cleave_rate_log 1 1



