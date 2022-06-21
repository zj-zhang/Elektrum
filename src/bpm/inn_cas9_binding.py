#!/usr/bin/env python

"""@package docstring
File: inn_cas9_binding.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description:
"""

import numpy as np

softmax_weights = np.array([[1, 0, 1, 0, 1, 0, 0],
                            [1, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 1, 0, 1, 0],
                            [1, 0, 0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 1, 0, 0],
                            [0, 1, 1, 0, 0, 1, 0],
                            [0, 1, 0, 1, 1, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0],
                            [1, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 1],
                            [0, 1, 1, 0, 0, 0, 1],
                            [0, 1, 0, 1, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 0, 1, 1],
                            [0, 1, 0, 0, 1, 0, 1],
                            [0, 1, 0, 0, 0, 1, 1],
                            [0, 0, 1, 0, 1, 0, 1],
                            [0, 0, 1, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1, 0, 1],
                            [0, 0, 0, 1, 0, 1, 1],
                            ])  # this works


##########################################
if __name__ == "__main__":
    print("Not implemented yet")
