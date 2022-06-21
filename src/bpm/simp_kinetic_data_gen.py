#!/usr/bin/env python

"""@package docstring
File: simp_kinetic_data_gen.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description:
"""


import re
import time
import yaml
from pprint import pprint
from pathlib import Path
import h5py

# Data manipulation
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.integrate import quad

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# chosen pwm for the negative log of kinetic rates
w1j = np.asarray([[-1., 1.],
                  [.5, -2.0]])

w2j = np.asarray([[1., .5],
                  [.5, -1.0]])

w3j = np.asarray([[.5, -.5],
                  [-1.5, 1.0]])

w4j = np.asarray([[1., .5],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [.5, -1.0]])

wij = [w1j, w2j, w3j, w4j]


# one-hot encoded input vector example
x_vec = np.asarray([[0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [1, 0]])


def get_activity(x_vec, wij):
    """TODO: Docstring for get_activity.

    @param x_vec TODO
    @param wij TODO
    @return: TODO

    """
    k1 = np.exp(np.tensordot(wij[0], x_vec[:2]))
    kminus1 = np.exp(np.tensordot(wij[1], x_vec[:2]))
    k2 = np.exp(np.tensordot(wij[2], x_vec[2:4]))
    k3 = np.exp(np.tensordot(wij[3], x_vec[:]))
    return k3 * k1 * k2 / (k2 * k3 + kminus1 * k3 + k1 * k3 + k1 * k2)


def activity_to_edit_exp(t, a=.5):
    return 1. - np.exp(- t / a)


def generate_pd_data(num_seq, seq_len=5):
    """TODO: Docstring for generate_pd_data.
    @return: TODO

    """
    # Generate random sequences
    seq_arr = np.random.randint(2, size=(num_seq, seq_len))

    # Create encoder
    ohe_enc = OneHotEncoder(sparse=False)
    ohe_enc.fit([0], [1])

    act_arr = np.zeros((num_seq))
    edit_prob_arr = np.zeros((num_seq))

    for i, seq in enumerate(seq_arr):
        seq_ohe = ohe_enc.transform(seq.reshape(-1, 1))
        act_arr[i] = get_activity(seq_ohe, wij)
        edit_prob_arr[i] = activity_to_edit_exp(act_arr[i])

    df = pd.DataFrame(np.stack((seq_arr, act_arr, edit_prob_arr), axis=-1))
    return df


##########################################
if __name__ == "__main__":
    t = get_activity(x_vec, wij)
    print(t)
