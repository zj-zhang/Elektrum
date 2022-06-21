#!/usr/bin/env python

"""@package docstring
File: calc_seq_bind_energy.py
Author: Adam Lamson
Email: alamson@flatironinstitute.org
Description: Energies and theory from
A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics
John SantaLucia
Proceedings of the National Academy of Sciences Feb 1998, 95 (4) 1460-1465; DOI: 10.1073/pnas.95.4.1460
Using the 'Unified' energies set
"""

import re
import numpy as np

INIT_PAIR_ENERGY = {
    'G': .98, 'C': .98,
    'A': 1.03, 'T': 1.03,
}

COMP_PAIR = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C',
    'N' : 'N'
}

# Units of kcal/mol and at 37C
# Watson-Crick NNs that have only two types of nucleotides have different
# energies depending on which comes first.
SEQ_53_PAIR_ENERGY = {
    'AA': -1.00, 'TT': -1.00,
    'AT': -.88,
    'TA': -.58,
    'CA': -1.45, 'TG': -1.45,  # The reverse and complementary = same energy
    'GT': -1.44, 'AC': -1.44,
    'CT': -1.28, 'AG': -1.28,
    'GA': -1.30, 'TC': -1.30,
    'CG': -2.17,
    'GC': -2.24,
    'GG': -1.84, 'CC': -1.84
}

# Units of kcal/mol (switch complementary pairs of 53 pair energies)
# It is as if you are reading sequence backwards
SEQ_35_PAIR_ENERGY = {
    'AA': -1.00, 'TT': -1.00,
    'TA': -.88,
    'AT': -.58,
    'GT': -1.45, 'AC': -1.45,
    'CA': -1.44, 'TG': -1.44,
    'GA': -1.28, 'TC': -1.28,
    'CT': -1.30, 'AG': -1.30,
    'GC': -2.17,
    'CG': -2.24,
    'GG': -1.84,
    'CC': -1.84
}

# Free energy if the sequence is self-complementary
ENERGY_SYM = .43


def is_self_comp(seq_str):
    """Is a sequence self-complementary?

    @param seq_str DNA sequence string
    @return: bool True if sequence is self-complementary

    >>> is_self_comp('CTAG')
    True
    >>> is_self_comp('CGTTGA')
    False

    """
    comp_seq = "".join([COMP_PAIR[n] for n in reversed(seq_str)])
    return comp_seq == seq_str


def create_pair_list(seq_str):
    """Returns a list of WC NNs from a sequence string

    @param seq_str DNA sequence string
    @return: list of consecutive pairs

    """
    return [n + seq_str[i + 1] for i, n in enumerate(seq_str[:-1])]


def get_seq_bind_energy_list(seq_str, dir_53_flag=True):
    """Calculates the free energy of two complementary ssDNA strands forming
    a dsDNA helix

    @param seq_str DNA sequence ex. 'ACTTAGATCCC'
    @param dir_53_flag Is the sequence given in the 5'-3' order
    @return: List of Gibbs free energy changes in 5'-e' order

    Sequence with known binding energy
    >>> get_seq_bind_energy_list('CGTTGA')
    [0.98, -2.17, -1.44, -1.0, -1.45, -1.3, 1.03]

    >>> get_seq_bind_energy_list("".join(reversed('CGTTGA')), dir_53_flag=False)
    [0.98, -2.17, -1.44, -1.0, -1.45, -1.3, 1.03]

    Self-complementary sequence
    >>> get_seq_bind_energy_list('CTAG')
    [0.98, -1.28, -0.58, -1.28, 0.98, 0.43]

    """
    # TODO: Add check to make sure sequence is valid <27-04-21, ARL> #
    # Return an average of everything if nn is undefined / N. 20210429, FZZ
    bind_energy_list = [INIT_PAIR_ENERGY[seq_str[0]] if seq_str[0] in INIT_PAIR_ENERGY else np.mean(list(INIT_PAIR_ENERGY.values())) ]
    nn_list = create_pair_list(seq_str)
    nn_energy_dict = SEQ_53_PAIR_ENERGY if dir_53_flag else SEQ_35_PAIR_ENERGY

    for nn in nn_list:
        # Return an average of everything if nn is undefined / N. 20210429, FZZ
        bind_energy_list += [nn_energy_dict[nn] if nn in nn_energy_dict else np.mean(list(nn_energy_dict.values()))]

    bind_energy_list += [INIT_PAIR_ENERGY[seq_str[-1]] if seq_str[-1] in INIT_PAIR_ENERGY else np.mean(list(INIT_PAIR_ENERGY.values())) ]

    bind_energy_list += [ENERGY_SYM] if is_self_comp(seq_str) else []

    return bind_energy_list if dir_53_flag else list(
        reversed(bind_energy_list))


def get_seq_bind_energy(seq_str, dir_53_flag=True):
    """Calculates the free energy of two complementary ssDNA strands forming
    a dsDNA helix

    @param seq_str DNA sequence ex. 'ACTTAGATCCC'
    @param dir_53_flag Is the sequence given in the 5'-3' order
    @return: List of Gibbs free energy changes

    Sequence with known binding energy
    >>> get_seq_bind_energy('CGTTGA')
    -5.35

    Self-complementary sequence
    >>> round(get_seq_bind_energy('CTAG'), 2)
    -0.75

    """
    be_list = get_seq_bind_energy_list(seq_str, dir_53_flag)
    return sum(be_list)


##########################################
if __name__ == "__main__":
    # print(get_seq_bind_energy('CGTTGA'))
    # print(is_self_comp('CTAG'))
    print(get_seq_bind_energy('CTAG'))
