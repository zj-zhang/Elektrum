#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AMBER NAS for incorporating KINN of different state numbers, and sequence context-specific effects
"""

# FZZ, 2022/10/16

import tensorflow as tf
import numpy as np
import os
import pickle
import h5py
from src.neural_network_builder import KineticNeuralNetworkBuilder
from src.kinetic_model import KineticModel
from amber.modeler.dag import get_layer
from amber.modeler import ModelBuilder

from amber import Amber
from amber.utils import run_from_ipython
from amber.architect import ModelSpace, Operation
import copy
import numpy as np
import scipy.stats as ss
from tensorflow.keras.optimizers import Adam
import argparse

# since tf 1.15 does not have multi-head attention officially implemented..
# we use this workaround
from keras_multi_head import MultiHeadAttention


class KinnLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        kinn_dir,
        manager_kws,
        kinn_trainable=False,
        channels=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kinn_trainable = kinn_trainable
        self.kinn_dir = kinn_dir
        self.manager_kws = manager_kws
        self.channels = channels
        self.kinn_layers = {}
        self.mb = None
        # get default session to feed model builder
        self.session = tf.keras.backend.get_session()

    def get_config(self):
        config = super(KinnLayer, self).get_config()
        config.update(
            {
                "kinn_dir": self.kinn_dir,
                "kinn_trainable": self.kinn_trainable,
                "manager_kws": self.manager_kws,
                "channels": self.channels,
            }
        )
        return config

    def build(self, input_shape):
        assert (
            isinstance(input_shape, (tuple, list)) and len(input_shape) == 2
        ), TypeError(
            "Expect a list of (hidden, input); got %s for kinn input shape"
            % input_shape
        )
        super().build(input_shape=input_shape)
        # will create two tensors with the same names..
        # mb = reload_from_dir(wd=self.kinn_dir, manager_kwargs=self.manager_kws, model_fn=KineticNeuralNetworkBuilder, sess=self.session)
        # self.mb = mb
        # self.kinn_layers =  {l.name: l for l in self.mb.model.layers}
        # self.kinn_header = tf.keras.models.Model(inputs=mb.model.inputs, outputs=self.kinn_layers['gather_rates'].output)

        n_channels = self.manager_kws.get("n_channels", 9)
        n_feats = self.manager_kws.get("n_feats", 25)
        replace_conv_by_fc = self.manager_kws.get("replace_conv_by_fc", False)
        output_op = self.manager_kws.get("output_op", None)
        with open(
            os.path.join(self.kinn_dir, "AmberSearchBestModel_config.pkl"), "rb"
        ) as f:
            model_params = pickle.load(f)
        self.bp = KineticModel(model_params)
        self.mb = KineticNeuralNetworkBuilder(
            kinn=self.bp,
            session=self.session,
            output_op=output_op,
            n_feats=n_feats,
            n_channels=n_channels,
            replace_conv_by_fc=replace_conv_by_fc,
        )

        # returns hidden layers in a dict
        ret = self.mb.build(return_intermediate=True)
        self.kinn_layers = ret
        self.kinn_header = tf.keras.models.Model(
            inputs=[
                self.kinn_layers["inputs_op"][j] for j in self.kinn_layers["inputs_op"]
            ],
            outputs=self.kinn_layers["gather_rates"],
        )
        self.kinn_header.load_weights(
            os.path.join(self.kinn_dir, "AmberSearchBestModel.h5")
        )
        self.output_op = output_op

        # king-altman constants
        self.king_altman_const = tf.constant(
            self.mb.kinn.get_ka_pattern_mat().transpose(), dtype=tf.float32
        )

        self.rate_contrib_map = []
        # rate_contrib_mat = (n_king_altman, n_rates)
        rate_contrib_mat = self.mb.kinn.get_rate_contrib_matrix()
        for k in range(rate_contrib_mat.shape[1]):
            # get each column as mask
            mask = rate_contrib_mat[:, k]
            assert np.sum(mask) <= 1, "k=%i, mask error for %s" % (k, mask)
            if np.sum(mask) == 0:
                continue
            rate_index = np.where(mask == 1)[0][0]
            self.rate_contrib_map.append((k, rate_index))
        # assert len(self.rate_contrib_map) == 1
        self.units = rate_contrib_mat.shape[0]
        self.lin_transform = tf.keras.layers.Dense(
            units=self.units,
            activation="linear",
            kernel_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-3),
            activity_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=1e-6),
            input_shape=(input_shape[0][-1],),
        )

        # check for kinn trainability
        for layer in self.kinn_header.layers:
            layer.trainable = self.kinn_trainable

    def kinn_body(self, rates):
        king_altman = tf.nn.softmax(tf.matmul(rates, self.king_altman_const))
        k = [x[0] for x in self.rate_contrib_map]
        rate_index = [x[1] for x in self.rate_contrib_map]
        rate_layer = tf.math.exp(tf.gather(rates, rate_index, axis=-1))
        ka_slice = tf.gather(king_altman, k, axis=-1)
        activity = tf.reduce_prod(rate_layer * ka_slice, axis=-1, keepdims=True)
        output = get_layer(x=activity, state=self.output_op, with_bn=False)
        return output

    def call(self, inputs):
        hidden, seq_ohe = inputs[0], inputs[1]

        # convert input to delta by linear transformation
        delta = self.lin_transform(hidden)

        # gather target channels, if necessary
        if self.channels is not None:
            seq_ohe = tf.gather(seq_ohe, self.channels, axis=-1)
        inp_list = self.mb.blockify_seq_ohe(seq_ohe)
        rates = self.kinn_header(inp_list)
        rates = rates + delta

        # kinn_body is implemented in forward pass
        output = self.kinn_body(rates)
        return output


class InceptionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_sizes, dilation_rates=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        assert len(self.filters) == len(self.kernel_sizes), ValueError(
            "different lengths for filters and kernel-sizes"
        )
        if dilation_rates is None:
            self.dilation_rates = [1 for _ in self.filters]
        else:
            self.dilation_rates = dilation_rates
            assert len(self.filters) == len(self.dilation_rates), ValueError(
                "different lengths for filters and dilation"
            )

    def get_config(self):
        config = super(InceptionLayer, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_sizes": self.kernel_sizes,
                "dilation_rates": self.dilation_rates,
            }
        )
        return config

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.conv_branches = []
        for f, ks, d in zip(*[self.filters, self.kernel_sizes, self.dilation_rates]):
            self.conv_branches.append(
                tf.keras.layers.Conv1D(
                    filters=f,
                    kernel_size=ks,
                    padding="same",
                    activation="relu",
                    dilation_rate=d,
                    input_shape=input_shape,
                )
            )

    def call(self, inputs):
        convs = [conv_op(inputs) for conv_op in self.conv_branches]
        layer_out = tf.concat(convs, axis=-1)
        return layer_out


class AttentionPooling(tf.keras.layers.Layer):
    """Applies multihead self-attention to the patches extracted form the
    trunk.

    Args:
        num_heads: The number attention heads.
        dropout: Dropout after attention
        flatten_op: Reshape operation.

    Inputs:
        Structured patches from the trunk.

    Outputs:
        Flatten patches after self-attention

    """

    def __init__(self, num_heads=2, dropout=0.2, flatten_op="flatten", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        assert flatten_op in ("flatten", "gap"), ValueError(
            "flatten_op must be in ('flatten', 'gap')"
        )
        self.flatten_op = flatten_op

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dropout": self.dropout,
                "num_heads": self.num_heads,
                "flatten_op": self.flatten_op,
            }
        )
        return config

    def build(self, input_shape):
        try:
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, dropout=self.dropout,
            )
            self.attn_type = "tf"
        except AttributeError:
            self.attention = tf.keras.models.Sequential(
                [
                    MultiHeadAttention(
                        head_num=self.num_heads, input_shape=input_shape
                    ),
                    tf.keras.layers.Dropout(self.dropout),
                ]
            )
            self.attn_type = "keras-multi-head"
        if self.flatten_op == "flatten":
            self.flatten = tf.keras.layers.Flatten()
        elif self.flatten_op == "gap":
            self.flatten = tf.keras.layers.GlobalAveragePooling1D()
        else:
            raise Exception()
        super().build(input_shape=input_shape)

    def call(self, x):
        if self.attn_type == "tf":
            out = self.attention([x, x])
        elif self.attn_type == "keras-multi-head":
            out = self.attention(x)
        else:
            raise Exception()
        out = self.flatten(out)
        return out


def BiDirectional(**kwargs):
    units = kwargs.get("units", 8)
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units, return_sequences=True), name="BiDir%i" % units
    )


def get_kinn_output_node():
    return tf.keras.layers.Lambda(
        lambda x: tf.math.log(tf.clip_by_value(x, 10 ** -7, 10 ** -1)) / np.log(10),
        name="output",
    )


def get_model_space_kinn():
    # Setup and params.
    state_space = ModelSpace()
    default_params = {}
    base_filter = 64
    param_list = [
        # Block 1:
        [
            {
                "filters": base_filter,
                "kernel_size": 1,
                "activation": "relu",
                "padding": "same",
                "name": "conv11",
            },
            {
                "filters": base_filter,
                "kernel_size": 3,
                "activation": "relu",
                "padding": "same",
                "name": "conv13",
            },
            {
                "filters": base_filter,
                "kernel_size": 3,
                "activation": "relu",
                "dilation_rate": 4,
                "padding": "same",
                "name": "conv13d4",
            },
            {
                "filters": base_filter,
                "kernel_size": 5,
                "activation": "relu",
                "padding": "same",
                "name": "conv15",
            },
        ],
        # Block 2:
        [
            {
                "filters": base_filter * 2,
                "kernel_size": 1,
                "activation": "relu",
                "padding": "same",
                "name": "conv21",
            },
            {
                "filters": base_filter * 2,
                "kernel_size": 3,
                "activation": "relu",
                "padding": "same",
                "name": "conv23",
            },
            {
                "filters": base_filter * 2,
                "kernel_size": 3,
                "activation": "relu",
                "dilation_rate": 4,
                "padding": "same",
                "name": "conv23d4",
            },
            {
                "filters": base_filter * 2,
                "kernel_size": 5,
                "activation": "relu",
                "padding": "same",
                "name": "conv25",
            },
        ],
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
            conv_states.append(Operation("conv1d", **d))

        # Add Inception Layer; reduce the filter number for each branch to match complexity
        conv_states.append(
            Operation(
                "InceptionLayer",
                filters=[base_filter * (2 ** (conv_seen)) // 4 for _ in range(4)],
                kernel_sizes=[1, 3, 3, 5],
                dilation_rates=[1, 1, 4, 1],
                name=f"inception_{conv_seen}",
            )
        )

        # Add a zero Layer
        if conv_seen > 0:
            conv_states.append(Operation("identity"))
        else:
            conv_states.append(
                Operation(
                    "conv1d",
                    activation="linear",
                    filters=base_filter,
                    kernel_size=1,
                    name="conv1_lin",
                )
            )

        state_space.add_layer(conv_seen * 2, conv_states)
        if i > 0:
            layer_embedding_sharing[conv_seen * 2] = 0
        conv_seen += 1

        # Add pooling states, if is the last conv.
        if i == len(param_list) - 1:
            bidirectional = [
                Operation("BiDirectional", units=32),
                Operation("BiDirectional", units=16),
                Operation("Identity"),
            ]
            state_space.add_layer(conv_seen * 2 - 1, bidirectional)
            pool_states = [
                Operation("Flatten"),
                Operation(
                    "AttentionPooling",
                    flatten_op="flatten",
                    dropout=0.2,
                    name="AtnFlat",
                ),
                # Operation('GlobalAvgPool1D'),
                # Operation('AttentionPooling', flatten_op='gap', dropout=0.1, name='AtnGap')
            ]
            state_space.add_layer(conv_seen * 2, pool_states)
        else:
            # Add dropout
            state_space.add_layer(
                conv_seen * 2 - 1,
                [
                    Operation("Identity"),
                    Operation("Dropout", rate=0.1),
                    Operation("Dropout", rate=0.3),
                    Operation("Dropout", rate=0.5),
                ],
            )
            if i > 0:
                layer_embedding_sharing[conv_seen * 2 - 1] = 1

    # Add an optional Dense layer
    state_space.add_layer(
        conv_seen * 2 + 1,
        [
            Operation("Dense", units=64, activation="relu"), 
            Operation("Identity"),
        ],
    )

    # Add final KINN layer.
    state_space.add_layer(
        conv_seen * 2 + 2,
        [
            # state 4
            Operation(
                "KinnLayer",
                kinn_dir="outputs/2022-05-21/KINN-wtCas9_cleave_rate_log-finkelstein-0-rep4-gRNA1/",
                manager_kws={"output_op": get_kinn_output_node},
                channels=np.arange(4, 13),
                name="kinn_f41",
            ),
            Operation(
                "KinnLayer",
                kinn_dir="outputs/2022-05-21/KINN-wtCas9_cleave_rate_log-finkelstein-0-rep5-gRNA2/",
                manager_kws={"output_op": get_kinn_output_node},
                channels=np.arange(4, 13),
                name="kinn_f42",
            ),
            # state 5
            Operation(
                "KinnLayer",
                kinn_dir="outputs/2022-05-30/KINN-wtCas9_cleave_rate_log-uniform-5-rep2-gRNA1/",
                manager_kws={"output_op": get_kinn_output_node},
                channels=np.arange(4, 13),
                name="kinn_u51",
            ),
            Operation(
                "KinnLayer",
                kinn_dir="outputs/2022-05-30/KINN-wtCas9_cleave_rate_log-uniform-5-rep3-gRNA2/",
                manager_kws={"output_op": get_kinn_output_node},
                channels=np.arange(4, 13),
                name="kinn_u52",
            ),
            # state 6
            Operation(
                "KinnLayer",
                kinn_dir="outputs/2022-05-30/KINN-wtCas9_cleave_rate_log-uniform-6-rep2-gRNA1/",
                manager_kws={"output_op": get_kinn_output_node},
                channels=np.arange(4, 13),
                name="kinn_u61",
            ),
            Operation(
                "KinnLayer",
                kinn_dir="outputs/2022-05-30/KINN-wtCas9_cleave_rate_log-uniform-6-rep2-gRNA2/",
                manager_kws={"output_op": get_kinn_output_node},
                channels=np.arange(4, 13),
                name="kinn_u62",
            ),
        ],
    )
    return state_space, layer_embedding_sharing


class TransferKinnModelBuilder(ModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.input_node = inputs_op[0]
        self.output_node = output_op[0]
        self.model_space = model_space
        self.custom_objects = {
            "InceptionLayer".lower(): InceptionLayer,
            "KinnLayer".lower(): KinnLayer,
            "AttentionPooling".lower(): AttentionPooling,
            "BiDirectional".lower(): BiDirectional,
        }

    def __call__(self, model_states):
        assert self.model_space is not None
        inp = get_layer(None, self.input_node, custom_objects=self.custom_objects)
        x = inp
        for i, state in enumerate(model_states):
            if issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
                op = self.model_space[i][state]
            elif isinstance(state, Operation) or callable(state):
                op = state
            else:
                raise Exception(
                    "cannot understand %s of type %s" % (state, type(state))
                )
            # if is KinnLayer, additionally connect the input op to here
            if op.Layer_type == "kinnlayer":  # no easy way right now
                x = get_layer([x, inp], op, custom_objects=self.custom_objects)
            else:
                x = get_layer(x, op, custom_objects=self.custom_objects)

        out = get_layer(x, self.output_node, custom_objects=self.custom_objects)
        model = tf.keras.models.Model(inputs=inp, outputs=out)
        model_compile_dict = copy.deepcopy(self.model_compile_dict)
        opt = model_compile_dict.pop("optimizer")()
        metrics = [
            x() if callable(x) else x for x in model_compile_dict.pop("metrics", [])
        ]
        model.compile(optimizer=opt, metrics=metrics, **model_compile_dict)
        return model


def amber_app(wd, run=False):
    # First, define the components we need to use
    with h5py.File("data/inVivoData.newValidSplit.h5", "r") as store:
        train_data = store.get("train")["x"][()], store.get("train")["y"][()]
        valid_data = store.get("valid")["x"][()], store.get("valid")["y"][()]
    # unpack data tuple
    (x_train, y_train), (x_valid, y_valid) = train_data, valid_data
    type_dict = {
        "controller_type": "GeneralController",
        "knowledge_fn_type": "zero",
        "reward_fn_type": "LossAucReward",
        # FOR RL-NAS
        "modeler_type": TransferKinnModelBuilder,
        "manager_type": "GeneralManager",
        "env_type": "ControllerTrainEnv",
    }

    # Next, define the specifics
    os.makedirs(wd, exist_ok=True)

    input_node = [Operation("input", shape=(25, 13), name="input")]

    output_node = [
        Operation(
            "dense",
            units=1,
            activation="sigmoid",
            name="output_final",
            kernel_constraint=tf.keras.constraints.NonNeg(),
            bias_constraint=tf.keras.constraints.NonNeg(),
            # XXX: this is super important!! if init values are clipped below zero, grads will diminish from sigmoid. FZZ 20221027
            kernel_initializer=tf.keras.initializers.Constant(1.75),
            bias_initializer=tf.keras.initializers.Constant(3.75),
        )
    ]

    model_space, layer_embedding_sharing = get_model_space_kinn()
    with open(os.path.join(wd, "model_space.pkl"), "wb") as f:
        pickle.dump((model_space, layer_embedding_sharing), f)
    batch_size = 25000
    use_ppo = False

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005,
        decay_steps=int(1000000 / batch_size) * 10,  # decay lr every 30 epochs
        decay_rate=0.9,
        staircase=False,
    )
    model_compile_dict = {
        "loss": "binary_crossentropy",
        "optimizer": lambda: Adam(learning_rate=lr_schedule),
        "metrics": ["acc", lambda: tf.keras.metrics.AUC(curve="PR")],
    }

    specs = {
        "model_space": model_space,
        "controller": {
            "share_embedding": layer_embedding_sharing,
            "with_skip_connection": False,
            "skip_weight": None,
            "lstm_size": 32,
            "lstm_num_layers": 1,
            "kl_threshold": 0.005,
            "train_pi_iter": 50 if use_ppo else 10,
            "optim_algo": "adam",
            "rescale_advantage_by_reward": False,
            "temperature": 2.0,
            "tanh_constant": 1.5,
            "buffer_size": 5,  # FOR RL-NAS
            "batch_size": 3,
            "use_ppo_loss": use_ppo,
        },
        "model_builder": {
            "batch_size": batch_size,
            "inputs_op": input_node,
            "outputs_op": output_node,
            "model_compile_dict": model_compile_dict,
        },
        "knowledge_fn": {"data": None, "params": {}},
        "reward_fn": {"method": "aupr", "batch_size": batch_size,},
        "manager": {
            "data": {
                "train_data": (x_train, y_train),
                "validation_data": (x_valid, y_valid),
            },
            "params": {
                "epochs": 20,
                "fit_kwargs": {
                    "earlystop_patience": 5,
                    # "class_weight": {0:1., 1:10.}
                },
                "predict_kwargs": {"batch_size": batch_size},
                "child_batchsize": batch_size,
                "store_fn": "model_plot",
                "working_dir": wd,
                "verbose": 0,
            },
        },
        "train_env": {
            "max_episode": 100,
            "max_step_per_ep": 5,
            "working_dir": wd,
            "time_budget": "72:00:00",
            "with_skip_connection": False,
            "save_controller_every": 1,
        },
    }

    # finally, run program
    amb = Amber(types=type_dict, specs=specs)
    if run:
        amb.run()
    return amb


if __name__ == "__main__":
    if not run_from_ipython():
        parser = argparse.ArgumentParser(
            description="Script for AMBER-search of Single-task runner"
        )
        parser.add_argument("--wd", type=str, help="working directory")

        args = parser.parse_args()
        amber_app(wd=args.wd, run=True)

# Or, run this in ipython terminal:
"""
%run src/transfer_learn

amb = amber_app(wd="outputs/test_tl_amber")
amb.manager.verbose = 1
#arc = [3,0,3,1,1,2,0,2]
#amb.manager.get_rewards(0, arc)
amb.run()

"""
