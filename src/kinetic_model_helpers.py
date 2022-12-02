#!/usr/bin/env python

import yaml
import random
from typing import *
from pathlib import Path
import numpy as np
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from copy import deepcopy


def make_encoders(values: List):
    """Generate label and one-hot encoder functions

    Parameters
    ----------
    values : list
        List of label values used to make one-hot encoder
        TODO: Give an example

    Returns
    -------
    LabelEncoder
        [description]
    OneHotEncoder
        [description]
    """
    # Create a label encoder that fits the values specified
    lab_enc = LabelEncoder()
    lab_enc.fit(values)
    tmp = lab_enc.transform(values)
    tmp = tmp.reshape(len(tmp), 1)
    # Create one hot encoder for the values in sequence
    one_enc = OneHotEncoder(sparse=False)
    _ = one_enc.fit(tmp)
    return lab_enc, one_enc


def gen_pos_weight_mat(guide_seq, seq_range, ind_scale=[1., -1.],
                       label_values=["A", "C", "G", "T"]):
    """ Make a position weight matrix contribution to rate based on template sequence given

    Parameters
    ----------
    template_str : str
        String of the guide sequence
    seq_range : list
        Indices that specify the sequence range that contribute to rate
    ind_scale : list, optional
        When matching the correct label, the nucleotide contributes the first
        index and when not matching it contributes the second,
        by default [1., -1.]
    nuc_order : list, optional
        Labels of the 'nucleotide' labels, by default ["A", "C", "G", "T"]

    Returns
    -------
    2D numpy.ndarray
        Position weight matrix that contributes to kinetic rate

    >>> gen_pos_weight_mat('TCGGTAGGATCGTAAGATAGTATT', [1, 6], ind_scale=[1.0, -1.0])
    array([[-1.,  1., -1., -1.],
           [-1., -1.,  1., -1.],
           [-1., -1.,  1., -1.],
           [-1., -1., -1.,  1.],
           [ 1., -1., -1., -1.]])
    """
    assert(guide_seq)
    template_seq = list(guide_seq)[seq_range[0]:seq_range[1]]

    lab_enc, one_enc = make_encoders(label_values)
    tmp = lab_enc.transform(template_seq)
    # one hot encode section of the guide sequence
    seq_ohe = one_enc.transform(tmp.reshape(-1, 1))
    # Set all weight values equal to lower contributing weight.
    #   Weight matrix has rows equal to the length of sequence range and
    #   columns equal to the number of label values
    weight_mat = np.repeat(
        np.asarray([[float(ind_scale[1])] * len(label_values)]),
        len(template_seq), axis=0)
    # Add back lower weight and higher contributing weight but only for
    # the nucleotides that match the templated string
    weight_mat += seq_ohe[...] * float(ind_scale[0] - ind_scale[1])
    return weight_mat


def nuc_distr(rate_dep_range, ind_scale):
    weight_mat = np.repeat(np.asarray([ind_scale]), rate_dep_range, axis=0)
    return weight_mat


def sigmoid(scale=1., trans=.0, amp=1.):
    def sigmoid_func(activity):
        return amp * (1. / (1. + np.exp(-(activity - trans) / scale)))
    return sigmoid_func
