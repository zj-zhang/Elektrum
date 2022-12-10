
import tensorflow as tf
import pytest
import numpy as np
import numpy.testing as npt
import os
from src.kinetic_model import KineticModel, KingAltmanKineticModel
from pathlib import Path


VALID_ADJ_MAT = np.array([[0, 1, 0, 1],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [1, 0, 1, 0]], dtype=float)

VALID_KINETIC_MAT_OCCUPANCY = np.array([[1, 1, 0, 1],
                                        [1, 1, 1, 0],
                                        [0, 1, 1, 1],
                                        [0, 0, 1, 1]])
VALID_KA_PATTERN_MAT = np.array([[0, 1, 0, 1, 0, 0, 1],
                                [0, 1, 0, 0, 1, 0, 1],
                                [0, 0, 1, 0, 1, 0, 1],
                                [0, 1, 0, 1, 0, 1, 0],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 0]], dtype=float)

KINETIC_MODELS_TO_TEST = [KineticModel(
    Path(__file__).resolve().parent / 'template_params.yaml')]

KA_KINETIC_MODELS_TO_TEST = [KingAltmanKineticModel(
    Path(__file__).resolve().parent / 'template_params.yaml')]


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


@pytest.mark.parametrize("k_model", KINETIC_MODELS_TO_TEST)
def test_kinetic_model_init(k_model):
    # Initialize from file
    check_matrix_occupancy(k_model.link_mat, VALID_ADJ_MAT)
    check_matrix_occupancy(k_model.kinetic_mat,
                           VALID_KINETIC_MAT_OCCUPANCY)
    # Initialize from yaml string

    # Initialize from dictionary

    # Initialize from neural network architecture

    pass


@pytest.mark.parametrize("k_model", KA_KINETIC_MODELS_TO_TEST)
def test_ka_kinetic_model_init(k_model):

    # Initialize from yaml string
    check_matrix_occupancy(k_model.link_mat, VALID_ADJ_MAT)
    check_matrix_occupancy(k_model.kinetic_mat, VALID_KINETIC_MAT_OCCUPANCY)
    npt.assert_array_equal(k_model.get_ka_pattern_mat(), VALID_KA_PATTERN_MAT)

    # Initialize from dictionary

    # Initialize from neural network architecture

    pass
# TODO Add checks for validity of models (number of states, connectedness, contribution rate)
