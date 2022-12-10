
import tensorflow as tf
import numpy as np
import os
from src.kinetic_model import KineticModel

TEMPLATE_TEST = ['T', 'C', 'G', 'G', 'T', 'A', 'G', 'G', 'A', 'T',
                 'C', 'G', 'T', 'A', 'A', 'G', 'A', 'T', 'A', 'G',
                 'T', 'A', 'T', 'T', 'C', 'A', 'G', 'G', 'A', 'C',
                 'C', 'C', 'C', 'G', 'T', 'T', 'A', 'A', 'C', 'C',
                 'A', 'T', 'T', 'T', 'C', 'G', 'A', 'A', 'A', 'G']
TEMPLATE_STR = "".join(TEMPLATE_TEST)

VALID_ADJ_SEQ_MAT = []
VALID_KINETIC_SEQ_MAT = []

# TODO paramertize this function
def test_seq_kinetic_model(arg):
    #TODO test 
    pass