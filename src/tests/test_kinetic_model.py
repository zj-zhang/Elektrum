
import tensorflow as tf
import numpy as np
import numpy.testing as npt
import os
from src.kinetic_model import KineticModel
from pathlib import Path


VALID_ADJ_MAT = np.array([[0, 1, 0, 1],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [1, 0, 1, 0]], dtype=float)

VALID_KINETIC_MAT_OCCUPANCY = np.array([[1, 1, 0, 1],
                                        [1, 1, 1, 0],
                                        [0, 1, 1, 1],
                                        [0, 0, 1, 1]])

# TODO paramertize this function


def turn_obj_mat_to_binary_mat(obj_mat):
    bin_mat = []
    for i, row in enumerate(obj_mat):
        bin_mat += [[]]
        for j, elem in enumerate(row):
            bin_mat[i] += [1] if elem else [0]
    return np.array(bin_mat)


def check_matrix_occupancy(kin_mat, known_kin_mat_ocp):
    kin_bin_mat = turn_obj_mat_to_binary_mat(kin_mat)
    npt.assert_array_equal(kin_bin_mat, known_kin_mat_ocp)


def test_kinetic_model_init():

    # Initialize from file
    yml_path = Path(__file__).resolve().parent / 'template_params.yaml'
    k_model = KineticModel(yml_path)

    # Initialize from yaml string
    npt.assert_array_equal(k_model.adj_mat, VALID_ADJ_MAT)
    check_matrix_occupancy(k_model.link_mat, VALID_ADJ_MAT)
    check_matrix_occupancy(k_model.kinetic_mat, VALID_KINETIC_MAT_OCCUPANCY)

    # Initialize from dictionary

    # Initialize from neural network architecture

    pass
