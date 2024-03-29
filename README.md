# Eleckrum
*Distilling the Physics from [Amber](https://github.com/zj-zhang/AMBER)*

This repo hosts the scripts and jupyter notebooks to run AMBER-based neural architecture search (NAS) for kinetic-interpretable neural networks
(KINN). Below we provide a brief summary/layout.

```
.
├── src                       # Package folder for running amber-nas and kinn
├── pub_notebooks             # Jupyter notebooks folder for publication figures
├── dev_notebooks             # Jupyter notebooks folder for developments
├── data                      # Input data folder can be downloaded using the link below
├── outputs                   # Output data folder generated by sbatch and notebooks
├── baselines                 # Folder for other machine-learning methods
├── sbatch_AmberCnn.sh        # Job script for amber-nas of deep CNN on CRISPR data
├── sbatch_AmberKinn.sh       # Job script for amber-nas of KINN on CRISPR data
├── sbatch_AmberSimKinn.sh    # Job script for amber-nas of KINN on synthetic data
├── submit.sh                 # Bash script for batch submissions of the above three jobs
└── jupyter.sh                # Job script for running notebooks
```

## Getting started

Elektrum is developed in Linux HPC environment with Python 3.7 and Keras 2.2.5/Tensorflow 1.15, with its software dependency managed by Anaconda.

We recommend you create a new conda environment for Elektrum. To start, use the following command:
```
conda create -n elektrum -c anaconda tensorflow-gpu=1.15.0 keras scikit-learn numpy~=1.18.5 h5py~=2.10.0 matplotlib seaborn
conda activate elektrum
pip install amber-automl==0.1.3 keras-multi-head==0.29.0 watermark
```

Upon successful creation, you will have a conda environment that satisfy all dependency requirements for Elektrum. Elektrum uses custom python scripts in `src` folder; no installation is needed.

The rest of this quick start manual will help you gather all necessary components to access our trained model for CRISPR/Cas9 off-target cleavage predictions and reproduce our published analysis. 

For this you will need to first download the data in Section "Download data and trained models", then follow instructions in "Reproducing the results". This is the best way to get familiar with this sophisticated model searching and building process and gear it towards you own use.

We also provide Jupyter notebooks and job scripts for re-training all models from scratch. This is outlined in section "Retraining from scratch/for your own data".

## Download data and trained models

Our searched KINN and Elektrum models with preprocessed datasets can be downloaded from this Google drive tarball:
https://drive.google.com/file/d/19a7U9W66O6m0huvMyZD9wmcmqpo1Nm_t/view?usp=sharing

The tarball takes ~185MB in local disk storage. You can download the above tarball using a python command-line utility [gdown](https://github.com/wkentaro/gdown), then untar it and move the results to corresponding folders:
```bash
gdown 19a7U9W66O6m0huvMyZD9wmcmqpo1Nm_t
md5sum Elektrum_0.0.3.tar.gz  # 8d9d5e42ee5ea8d5e579d7dd02af846c 
tar -xvzf Elektrum_0.0.3.tar.gz
mv tarball_0.0.3/* ./  # move data and output folders to root level
```

## Reproducing the results

We provide a series of Jupyter notebooks for analyzing and visualizing the results. With the downloaded data and trained models, this should run fast even if you don't have access to a GPU.

To reproduce our published figures, go to notebooks under the folder `pub_notebooks`. These notebooks are named by the corresponding Figure names. These have been thoroughly tested to run with our provided tarball data and model file. 

To understand the development and analysis throughout Elektrum, go to notebooks under the folder `dev_notebooks`. These notebooks are ordered numerically to reflect how Elektrum was developed. Some intermediate files may be missing because Elektrum went through several development phases. You can leave a Github Issue [here](https://github.com/zj-zhang/Elektrum/issues) to request missing files.

## Retraining from scratch/for your own data

We also provide the scripts for training new KINN and Elektrum models. These python scripts are located in `src/`. Because Elektrum is developed in HPC, we share our job submission scripts on Slurm to run multiple independent replicates.

For simuated kinetic data and CRISPR/Cas9 MPKP kinetic data, use `sbatch_AmberSimKinn.sh` and `sbatch_AmberKinn.sh` to build KINN. For building deep CNNs using the same kinetic data, use `sbatch_AmberCnn.sh`.

For our transfer learning-enhanced Elektrum model, use `sbatch_AmberTL.sh`. 

Each of these scripts will output their trained models and logs in the `./outputs/` folder.


## Citation
If you find Elektrum useful, please cite:

>Interpretable neural architecture search and transfer learning for understanding sequence dependent enzymatic reactions. Zijun Zhang†, Adam R Lamson†, Michael Shelley and Olga Troyanskaya. 

[To Be Updated]

## Contact
Still have questions? Leave a Github Issue [here](https://github.com/zj-zhang/Elektrum/issues) and I'll get back to you.

