sbatch -J wt-f0 sbatch.sh wtCas9_cleave_rate_log finkelstein 0 0
sbatch -J wt-f1 sbatch.sh wtCas9_cleave_rate_log finkelstein 1 1
#sbatch -J wt0 sbatch2.sh wtCas9_cleave_rate_log 0 0
#sbatch -J wt1 sbatch2.sh wtCas9_cleave_rate_log 1 1


# ON HOLD
#sbatch -J wt-u3 sbatch.sh wtCas9_cleave_rate_log uniform 3
#sbatch -J wt-u4 sbatch.sh wtCas9_cleave_rate_log uniform 4
#sbatch -J wt-u5 sbatch.sh wtCas9_cleave_rate_log uniform 5

#sbatch -J hf1-f sbatch.sh Cas9_HF1_cleave_rate_log finkelstein 4
#sbatch -J hf1-u3 sbatch.sh Cas9_HF1_cleave_rate_log uniform 3
#sbatch -J hf1-u4 sbatch.sh Cas9_HF1_cleave_rate_log uniform 4
#sbatch -J hf1-u5 sbatch.sh Cas9_HF1_cleave_rate_log uniform 5

#sbatch -J wt-u7 sbatch.sh wtCas9_cleave_rate_log uniform 7
#sbatch -J hf1-u7 sbatch.sh Cas9_HF1_cleave_rate_log uniform 7
#sbatch -J wt-u6 sbatch.sh wtCas9_cleave_rate_log uniform 6
#sbatch -J hf1-u6 sbatch.sh Cas9_HF1_cleave_rate_log uniform 6


