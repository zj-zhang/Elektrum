#!/usr/bin/env python
# coding: utf-8
# # Probablistic model building genetic algorithm

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from src.kinetic_model import KineticModel, modelSpace_to_modelParams
from src.neural_network_builder import KineticNeuralNetworkBuilder, KineticEigenModelBuilder
from src.neural_search import search_env
from src.data import load_finkelstein_data as get_data
from src.model_spaces import get_cas9_uniform_ms, get_cas9_finkelstein_ms, get_cas9_finkelstein_ms_with_hidden
from src.reload import reload_from_dir

import warnings
warnings.filterwarnings('ignore')

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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, choices="""wtCas9_cleave_rate_log
Cas9_enh_cleave_rate_log
Cas9_hypa_cleave_rate_log
Cas9_HF1_cleave_rate_log
wtCas9_cleave_rate_log_specificity
Cas9_enh_cleave_rate_log_specificity
Cas9_hypa_cleave_rate_log_specificity
Cas9_HF1_cleave_rate_log_specificity
wtCas9_ndABA
Cas9_enh_ndABA
Cas9_hypa_ndABA
Cas9_HF1_ndABA""".split(), required=True)
    parser.add_argument('--use-sink-state', action="store_true", default=False)
    parser.add_argument('--ms', type=str, choices=['finkelstein', 'uniform'], required=True)
    parser.add_argument('--wd', type=str, required=True)
    parser.add_argument('--n-states', type=int, default=4, required=False)
    parser.add_argument('--win-size', type=int, default=None, required=False)
    parser.add_argument("--switch", type=int, default=0, help="switch to train on gRNA2, test on gRNA1; default 0-false")

    args = parser.parse_args()
    os.makedirs(args.wd, exist_ok=True)
    pickle.dump(args, open(os.path.join(args.wd, "args.pkl"), "wb"))
    return args


def main():
    args = parse()
    if args.ms == "finkelstein":
        kinn_model_space = get_cas9_finkelstein_ms_with_hidden(use_sink_state=args.use_sink_state)
    else:
        kinn_model_space = get_cas9_uniform_ms(n_states=args.n_states, st_win_size=args.win_size, use_sink_state=args.use_sink_state)
    print("use sink state:", args.use_sink_state)
    print(kinn_model_space)
    controller = pmbga.ProbaModelBuildGeneticAlgo(
                model_space=kinn_model_space,
                buffer_type='population',
                buffer_size=50,  # buffer size controlls the max history going back
                batch_size=1,    # batch size does not matter in this case; all arcs will be retrieved
                ewa_beta=0.0     # ewa_beta approximates the moving average over 1/(1-ewa_beta) prev points
            )
    make_switch = args.switch != 0
    logbase = 10
    res = get_data(target=args.target, make_switch=make_switch, logbase=logbase, include_ref=False)
    print("switch gRNA_1 to testing and gRNA_2 to training:", make_switch)
    # unpack data tuple
    (x_train, y_train), (x_test, y_test) = res
    if args.use_sink_state:
        output_op = lambda: tf.keras.layers.Lambda(lambda x: tf.math.log(tf.clip_by_value(tf.reshape(- x[:,1], (-1,1)), 10**-5, 10**-1))/np.log(logbase), name="output_slice")
        #output_op = lambda: tf.keras.layers.Lambda(lambda x: tf.clip_by_value(tf.reshape(- x[:,1], (-1,1)), 10**-5, 10**-1), name="output_slice")
    else:
        output_op = lambda: tf.keras.layers.Lambda(lambda x: tf.math.log(tf.clip_by_value(x, 10**-5, 10**-1))/np.log(logbase), name="output_log")
        #output_op = lambda: tf.keras.layers.Dense(units=1, activation="linear", name="output_nonneg", kernel_constraint=tf.keras.constraints.NonNeg())
    # trainEnv parameters
    evo_params = dict(
        model_fn = KineticEigenModelBuilder if args.use_sink_state else KineticNeuralNetworkBuilder,
        samps_per_gen = 10,   # how many arcs to sample in each generation; important
        max_gen = 600,
        patience = 200,
        n_warmup_gen = 0,
        train_data = (x_train, y_train),
        test_data = (x_test, y_test)
    )
    # this learning rate is trickier than usual, for eigendecomp to work
    initial_learning_rate = 0.01
    batch_size = 512
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10*int(7000/batch_size), # decrease every 10 epochs
        decay_rate=0.9,
        staircase=True)

    manager_kwargs = {
        'output_op': output_op,
        'n_feats': 25,
        'n_channels': 9,
        'batch_size': batch_size,
        'epochs': 300,
        'earlystop': 15,
        'optimizer': lambda: tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        'verbose': 0
    }
    controller, hist, stat_df = search_env(
        neural_search_controller=controller,
        workdir=args.wd,
        evo_params=evo_params,
        manager_kwargs=manager_kwargs
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
    return controller



if __name__ == "__main__":
    if not amber.utils.run_from_ipython():
        main()

