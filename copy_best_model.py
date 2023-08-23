import sys
import os
import shutil
import pandas as pd

src_dir = sys.argv[1]
tar_dir = sys.argv[2]

print(f"going to copy {src_dir}/*/weights/ to {tar_dir}/*/weights, continue?")
cont = input("[y/n]: ")
if cont!='y': sys.exit()

def get_model_single_run(wd):
    n_tail = 25 # only for transfer learning
    train_hist = pd.read_table(os.path.join(wd, "train_history.csv"), sep=",", header=None)
    best_trial_id = train_hist.tail(n_tail).sort_values(2, ascending=False).head(1)[0]
    return best_trial_id

runs = [_ for _ in os.listdir(src_dir)]
for wd in runs:
    trial = get_model_single_run(os.path.join(src_dir,wd))
    src = os.path.join(src_dir, wd, 'weights', 'trial_%i'%trial)
    tar = os.path.join(tar_dir, wd, 'weights')
    os.makedirs(tar, exist_ok=True)
    print(src, tar)
    os.system(f"cp -r {src} {tar}/")
    os.system(f"cp {os.path.join(src_dir, wd)}/* {os.path.join(tar_dir, wd)}/")

