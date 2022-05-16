#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example wrapper `Amber` use for searching cas9 off-target kinetic prediction
FZZ, Jan 28, 2022
"""

from amber import Amber
from amber.utils import run_from_ipython, get_available_gpus
from amber.architect import ModelSpace, Operation
import sys
import os
import pickle
import copy
import numpy as np
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam
#from keras.optimizers import SGD, Adam
import argparse
import pickle
from src.data import load_finkelstein_data as get_data


def get_model_space_long():
    # Setup and params.
    state_space = ModelSpace()
    default_params = {}
    param_list = [
            # Block 1:
            [
                {"filters": 16, "kernel_size": 1, "activation": "relu", "padding": "valid"},
                {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "valid"},
                {"filters": 16, "kernel_size": 7, "activation": "relu", "padding": "valid"},
                {"filters": 16, "kernel_size": 1, "activation": "tanh", "padding": "valid"},
                {"filters": 16, "kernel_size": 3, "activation": "tanh", "padding": "valid"},
                {"filters": 16, "kernel_size": 7, "activation": "tanh", "padding": "valid"},
            ],
            # Block 2:
            [
                {"filters": 64, "kernel_size": 1, "activation": "relu", "padding": "valid"},
                {"filters": 64, "kernel_size": 3, "activation": "relu", "padding": "valid"},
                {"filters": 64, "kernel_size": 7, "activation": "relu", "padding": "valid"},
                {"filters": 64, "kernel_size": 1, "activation": "tanh", "padding": "valid"},
                {"filters": 64, "kernel_size": 3, "activation": "tanh", "padding": "valid"},
                {"filters": 64, "kernel_size": 7, "activation": "tanh", "padding": "valid"},
            ],
            # Block 3:
            [
                {"filters": 256, "kernel_size": 1, "activation": "relu", "padding": "valid"},
                {"filters": 256, "kernel_size": 3, "activation": "relu", "padding": "valid"},
                {"filters": 256, "kernel_size": 7, "activation": "relu", "padding": "valid"},
                {"filters": 256, "kernel_size": 1, "activation": "tanh", "padding": "valid"},
                {"filters": 256, "kernel_size": 3, "activation": "tanh", "padding": "valid"},
                {"filters": 256, "kernel_size": 7, "activation": "tanh", "padding": "valid"},

            ],
            # Block 4:
            #[
            #    {"filters": 256, "kernel_size": 1, "activation": "relu", "padding": "valid"},
            #    {"filters": 256, "kernel_size": 3, "activation": "relu", "padding": "valid"},
            #    {"filters": 256, "kernel_size": 7, "activation": "relu", "padding": "valid"},
            #    {"filters": 256, "kernel_size": 1, "activation": "tanh", "padding": "valid"},
            #    {"filters": 256, "kernel_size": 3, "activation": "tanh", "padding": "valid"},
            #    {"filters": 256, "kernel_size": 7, "activation": "tanh", "padding": "valid"},
            #],

        ]

    # Build state space.
    layer_embedding_sharing = {}
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = []
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(Operation('conv1d', name="conv{}".format(conv_seen), **d))
        if conv_seen > 0:
            conv_states.append(Operation('identity', name="id{}".format(conv_seen)))
        else:
            conv_states.append(Operation('conv1d', name="conv{}".format(conv_seen), activation="linear", filters=16, kernel_size=1))

        state_space.add_layer(conv_seen*2, conv_states)
        if i > 0:
            layer_embedding_sharing[conv_seen*2] = 0
        conv_seen += 1

        # Add pooling states, if is the last conv.
        if i == len(param_list) - 1:
            pool_states = [
                    Operation('Flatten'),
                    Operation('GlobalMaxPool1D'),
                    Operation('GlobalAvgPool1D')
                ]
            state_space.add_layer(conv_seen*2-1, pool_states)
        else:
            # Add dropout
            state_space.add_layer(conv_seen*2-1, [
                Operation('Identity'),
                Operation('Dropout', rate=0.1),
                Operation('Dropout', rate=0.3),
                Operation('Dropout', rate=0.5)
                ])
            if i > 0:
                layer_embedding_sharing[conv_seen*2-1] = 1

    # Add final classifier layer.
    state_space.add_layer(conv_seen*2, [
            Operation('Dense', units=64, activation='relu'),
            Operation('Dense', units=32, activation='relu'),
            Operation('Identity')
        ])
    return state_space, layer_embedding_sharing



def amber_app(wd, target="wtCas9_cleave_rate_log", make_switch=False, run=False):
    # First, define the components we need to use
    print("switch gRNA_1 to testing and gRNA_2 to training:", make_switch)
    x1_train, y1_train, x1_test, y1_test, x2_train, y2_train, x2_test, y2_test = get_data(target=target, make_switch=make_switch)
    type_dict = {
        'controller_type': 'GeneralController',
        'knowledge_fn_type': 'zero',
        'reward_fn_type': 'LossAucReward',

        # FOR RL-NAS
        'modeler_type': 'KerasModelBuilder',
        'manager_type': 'GeneralManager',
        'env_type': 'ControllerTrainEnv'
    }


    # Next, define the specifics
    os.makedirs(wd, exist_ok=True)

    input_node = [
            Operation('input', shape=(25,9), name="input")
            ]

    output_node = [
            Operation('dense', units=1, activation='linear', name="output")
            ]

    model_compile_dict = {
        'loss': 'mae',
        'optimizer': 'adam',
    }

    model_space, layer_embedding_sharing = get_model_space_long()
    batch_size = 768
    use_ppo = True

    specs = {
        'model_space': model_space,

        'controller': {
                'share_embedding': layer_embedding_sharing,
                'with_skip_connection': False,
                'skip_weight': None,
                'lstm_size': 64,
                'lstm_num_layers': 1,
                'kl_threshold': 0.01,
                'train_pi_iter': 50 if use_ppo else 10,
                'optim_algo': 'adam',
                'rescale_advantage_by_reward': False,
                'temperature': 2.0,
                'tanh_constant': 1.5,
                'buffer_size': 10,  # FOR RL-NAS
                'batch_size': 5,
                'use_ppo_loss': use_ppo
        },

        'model_builder': {
            'batch_size': batch_size,
            'inputs_op': input_node,
            'outputs_op': output_node,
            'model_compile_dict': model_compile_dict,
        },

        'knowledge_fn': {'data': None, 'params': {}},

        'reward_fn': {'method': lambda y_true, y_score: ss.pearsonr(y_true, y_score.flatten())[0]},

        'manager': {
            'data': {
                'train_data': (x1_train, y1_train),
                'validation_data': (x2_train, y2_train),
            },
            'params': {
                'epochs': 400,
                'fit_kwargs': {
                    'earlystop_patience': 30,
                    #'max_queue_size': 50,
                    #'workers': 3
                    },
                'child_batchsize': batch_size,
                'store_fn': 'model_plot',
                'working_dir': wd,
                'verbose': 0
            }
        },

        'train_env': {
            'max_episode': 350,
            'max_step_per_ep': 10,
            'working_dir': wd,
            'time_budget': "48:00:00",
            'with_skip_connection': False,
            'save_controller_every': 1
        }
    }


    # finally, run program
    amb = Amber(types=type_dict, specs=specs)
    if run:
        amb.run()
    return amb


if __name__ == '__main__':
    if not run_from_ipython():
        parser = argparse.ArgumentParser(description="Script for AMBER-search of Single-task runner")
        parser.add_argument("--wd", type=str, help="working directory")
        parser.add_argument("--switch", type=int, default=0, help="switch to train on gRNA2, test on gRNA1; default 0-false")
        parser.add_argument("--target", choices="""wtCas9_cleave_rate_log
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
Cas9_HF1_ndABA""".split(), default='wtCas9_cleave_rate_log', type=str, help="target to train")

        args = parser.parse_args()
        pickle.dump(args, open(os.path.join(args.wd, "args.pkl"), "wb"))
        make_switch = args.switch!=0
        amber_app(
                wd=args.wd,
                target=args.target,
                make_switch=make_switch,
                run=True
                )
