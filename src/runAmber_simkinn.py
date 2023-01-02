#!/usr/bin/env python
# coding: utf-8
# # Probablistic model building genetic algorithm

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from src.kinetic_model import KineticModel, modelSpace_to_modelParams
from src.neural_network_builder import KineticNeuralNetworkBuilder, KineticEigenModelBuilder
from src.neural_search import search_env
from src.model_spaces import get_sim_model_space as get_model_space
from src.reload import reload_from_dir

import warnings
warnings.filterwarnings('ignore')
import yaml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as  sns
import scipy.stats as ss
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os
import sys
import argparse
import pickle
import amber
print(amber.__version__)
from amber.architect import pmbga

def get_data(fp, logbase=10, noise_sigma=0):
    if fp.endswith("csv"):
        input_seqs_ohe, y_train, test_seqs_ohe, y_test = load_csv(fp)
    elif fp.endswith("tsv"):
        input_seqs_ohe, y_train, test_seqs_ohe, y_test = load_tsv(fp)
    else:
        raise Exception("filepath not understood: %s" % fp)
    if logbase is not None:
        y_train = np.log(y_train) / np.log(logbase)
        y_test = np.log(y_test) / np.log(logbase)
    if noise_sigma > 0:
        y_train += np.random.normal(0, noise_sigma, size=len(y_train))
        y_test += np.random.normal(0, noise_sigma, size=len(y_test))
    return (input_seqs_ohe, y_train), (test_seqs_ohe, y_test)


# data loaders for different formatted synthetic data
def load_csv(fp, output_act=True):
    # one-hot encoder
    x_set = [['A'], ['C'], ['G'], ['T']]
    enc = OneHotEncoder(sparse=False, categories='auto')
    _ = enc.fit(x_set)
    # read in data
    with open(fp, 'r') as f:
        data = pd.read_csv(f, index_col=0)
    data.dropna(inplace=True)
    data['seq_ohe'] = [enc.transform(
        [[t] for t in row[1][0:50]]) for row in data.iterrows()]
    if output_act is True:
        data['obs'] = data['50']
        data['obs'] /= data['obs'].max()
    else:
        data['obs'] = data['51']
    #print(data['obs'].describe())
    # split train-test
    gen_df = data
    X_train, X_test, y_train, y_test = train_test_split(
        gen_df['seq_ohe'].values, gen_df['obs'].values, test_size=0.2, random_state=777)
    input_seqs_ohe = []
    for i in range(len(X_train)):
        input_seqs_ohe += [X_train[i]]
    test_seqs_ohe = []
    for i in range(len(X_test)):
        test_seqs_ohe += [X_test[i]]

    input_seqs_ohe = np.array(input_seqs_ohe)
    test_seqs_ohe = np.array(test_seqs_ohe)
    return input_seqs_ohe, y_train, test_seqs_ohe, y_test


def load_tsv(fp):
    # one-hot encoder
    x_set = [['A'], ['C'], ['G'], ['T']]
    enc = OneHotEncoder(sparse=False, categories='auto')
    _ = enc.fit(x_set)

    # read in data
    data = pd.read_table(fp)
    data.dropna(inplace=True)
    data['seq_ohe'] = [enc.transform(
        [[t] for t in row['seq']]) for _, row in data.iterrows()]
    data['obs'] = - data['first_eigval']
    #print(data['obs'].describe())
    # split train-test
    gen_df = data
    X_train, X_test, y_train, y_test = train_test_split(
        gen_df['seq_ohe'].values, gen_df['obs'].values, test_size=0.2, random_state=777)
    input_seqs_ohe = []
    for i in range(len(X_train)):
        input_seqs_ohe += [X_train[i]]
    test_seqs_ohe = []
    for i in range(len(X_test)):
        test_seqs_ohe += [X_test[i]]
    input_seqs_ohe = np.array(input_seqs_ohe)
    test_seqs_ohe = np.array(test_seqs_ohe)
    return input_seqs_ohe, y_train, test_seqs_ohe, y_test


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-posterior', action="store_true", default=False)
    parser.add_argument('--patience', type=int, default=200, required=False)
    parser.add_argument('--max-gen', type=int, default=600, required=False)
    parser.add_argument('--samps-per-gen', type=int, default=5, required=False)
    parser.add_argument('--wd', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--param-file', type=str, required=True)
    parser.add_argument('--use-sink-state', action="store_true", default=False)
    parser.add_argument('--logbase', type=float, default=10)
    parser.add_argument('--noise-sigma', type=float, default=0)

    args = parser.parse_args()
    os.makedirs(args.wd, exist_ok=True)
    pickle.dump(args, open(os.path.join(args.wd, "args.pkl"), "wb"))
    return args


def main():
    args = parse()
    kinn_model_space = get_model_space(use_sink_state=args.use_sink_state)
    print("use sink state:", args.use_sink_state)
    print(kinn_model_space)
    controller = pmbga.ProbaModelBuildGeneticAlgo(
                model_space=kinn_model_space,
                buffer_type='population',
                buffer_size=50,  # buffer size controlls the max history going back
                batch_size=1,    # batch size does not matter in this case; all arcs will be retrieved
                ewa_beta=0.0     # ewa_beta approximates the moving average over 1/(1-ewa_beta) prev points
            )
    res = get_data(fp=args.data_file, 
                   logbase=args.logbase, 
                   noise_sigma=args.noise_sigma)
    (x_train, y_train), (x_test, y_test) = res
    
    logbase = args.logbase
    # TODO What is the difference between these too
    if args.use_sink_state:
        output_op = lambda: tf.keras.layers.Lambda(lambda x: 
            tf.math.log(tf.clip_by_value(tf.reshape(- x[:,1], (-1,1)), 10**-16, 10**3))/np.log(logbase), 
            name="output_slice")
    else:
        output_op = lambda: tf.keras.layers.Lambda(lambda x: tf.math.log(tf.clip_by_value(x, 10**-16, 10**3))/np.log(logbase), 
                                                   name="output_log")
    # trainEnv parameters
    evo_params = dict(
        model_fn = KineticEigenModelBuilder if args.use_sink_state else KineticNeuralNetworkBuilder,
        samps_per_gen = args.samps_per_gen,   # how many arcs to sample in each generation; important
        max_gen = args.max_gen,
        patience = args.patience,
        n_warmup_gen = 0,
        train_data = (x_train, y_train),
        test_data = (x_test, y_test)
    )
    # this learning rate is trickier than usual, for eigendecomp to work
    initial_learning_rate = 0.05
    batch_size = 2048
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10*int(20000/batch_size), # decrease every 10 epochs
        decay_rate=0.9,
        staircase=True)

    manager_kwargs = {
        'output_op': output_op,
        'n_feats': 50,
        'n_channels': 4,
        'batch_size': batch_size,
        'epochs': 20 if args.use_sink_state else 100,
        'earlystop': 5,
        'optimizer': lambda: tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        'verbose': 0
    }
    controller, hist, stat_df = search_env(
        neural_search_controller=controller,
        workdir=args.wd,
        evo_params=evo_params,
        manager_kwargs=manager_kwargs,
        disable_posterior_update=args.disable_posterior
    )
    # plot the best model
    mb = reload_from_dir(wd=args.wd, manager_kwargs=manager_kwargs, model_fn=evo_params['model_fn'])
    tf.keras.utils.plot_model(mb.model, to_file=os.path.join(args.wd, "model.png"))
    y_hat = mb.predict(x_test).flatten()
    h = sns.jointplot(y_test, y_hat)
    h.set_axis_labels("obs", "pred", fontsize=16)
    p = ss.pearsonr(y_hat, y_test)
    h.fig.suptitle("Testing prediction, pcc=%.3f"%p[0], fontsize=16)
    plt.savefig(os.path.join(args.wd, "test_pred.png"))
    
    # for sim data: analyze post model distr
    print("plot model posterior")
    plot_model_post_distr(controller=controller, args=args)
    return controller


def plot_model_post_distr(controller, args):
    # ground-truth params for synthetic data
    with open(args.param_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        kinn_gr = KineticModel(config)
   # START SITE
    fig, axs_ = plt.subplots(3,3, figsize=(15,15))
    axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
    for k in controller.model_space_probs:
        if k[-1] == 'RANGE_ST':
            try:
                d = controller.model_space_probs[k].sample(size=1000)
            except:
                continue
            ax = axs[k[0]]
            _ = sns.distplot(d, label="Post", ax=ax)
            _ = sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
            if k[0] < 7:
                _ = ax.axvline(x=kinn_gr.model_params['Rates'][k[0]]['input_range'][0],linestyle='--', color='grey')
                _ = ax.set_title(
                    ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d)), 
                              '\nGround truth', str(kinn_gr.model_params['Rates'][k[0]]['input_range'][0])])
                )
            else:
                _ = ax.set_title(
                    ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))

            #_ = ax.set_xlim(0,50)
    fig.tight_layout()
    fig.savefig(os.path.join(args.wd, "syn_range_st.png"))

    # CONV RANGE
    fig, axs_ = plt.subplots(3,3, figsize=(15,15))
    axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
    for k in controller.model_space_probs:
        if k[-1] == 'RANGE_D':
            d = controller.model_space_probs[k].sample(size=1000)
            ax = axs[k[0]]
            _ = sns.distplot(d, ax=ax)
            _ = sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
            if k[0] < 7:
                D = kinn_gr.model_params['Rates'][k[0]]['input_range'][1] - kinn_gr.model_params['Rates'][k[0]]['input_range'][0]
                _ = ax.axvline(x=D,linestyle='--', color='grey')
                _ = ax.set_title(
                    ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d)), '\nGround truth', str(D)])
                )
            else:
                _ = ax.set_title(
                    ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
            #_ = ax.set_xlim(0,20)    
    fig.tight_layout()
    fig.savefig(os.path.join(args.wd, "syn_range_d.png"))

    # EDGE PRESENCE
    fig, axs_ = plt.subplots(3,3, figsize=(15,15))
    axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
    for k in controller.model_space_probs:
        if k[-1] == 'EDGE':
            d = controller.model_space_probs[k].sample(size=1000)
            ax = axs[k[0]]
            sns.distplot(d, ax=ax)
            sns.distplot(controller.model_space_probs[k].prior_dist, ax=ax)
            ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
            #_ = ax.set_xlim(0,20)    
    fig.tight_layout()
    fig.savefig(os.path.join(args.wd, "syn_edge.png"))




if __name__ == "__main__":
    if not amber.utils.run_from_ipython():
        main()

