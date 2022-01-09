# %%
import yaml
from pathlib import Path
import numpy as np
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

def modelSpace_to_modelParams(model_arcs):
    """example config from yaml:
States:
  - '0' 
  - '1' 
  - '2'
  - '3'
Rates:
  - name: "k_{01}"
    state_list: ['0', '1']
    input_range: [5,10]

  - name: "k_{10}"
    state_list: ['1', '0']
    input_range: [10,15]
Data:
  contrib_rate_names: ['k_{30}']
"""
    model_params = {'States':set([]), 'Rates':[], 'Data':{'contrib_rate_names':[]}}
    for arc in model_arcs:
        if not arc.Layer_attributes.get('EDGE', True):
            continue
        s, t = arc.Layer_attributes['SOURCE'], arc.Layer_attributes['TARGET']
        inp_r = [arc.Layer_attributes['RANGE_ST'], arc.Layer_attributes['RANGE_ST']+arc.Layer_attributes['RANGE_D']]
        ks = arc.Layer_attributes['kernel_size']
        rate_name = "k_{}{}".format(s,t)
        model_params['States'].add(s)
        model_params['States'].add(t)
        tmp = {
            'name': rate_name,
            'state_list': [s, t],
            'input_range': inp_r,
            'kernel_size': ks
        }
        tmp.update(**arc.Layer_attributes)
        model_params['Rates'].append(tmp)
        if arc.Layer_attributes.get('CONTRIB', False):
            model_params['Data']['contrib_rate_names'].append(rate_name)
    model_params['States'] = list(model_params['States'])
    return model_params


def wang_algebra_sequences(branch_list):
    """Return a list of lists of all acceptable KA patterns checking
    for duplicate states.

    >>> print(wang_algebra_sequences([[2, 5], [5, 6], [7, 2]]))
    [[2, 5, 7], [2, 6, 7], [5, 6, 7], [5, 6, 2]]
    """
    assert(len(branch_list) > 0)
    if len(branch_list) == 1:
        return [[state] for state in branch_list[0]]

    seq = []
    sub_seq_list = wang_algebra_sequences(branch_list[1:])
    for sseq in sub_seq_list:
        for state in branch_list[0]:
            if state not in sseq:
                seq += [[state] + sseq]
    return seq


class RateMat():
    def __init__(self, params, template=None):
        self.__dict__ = params
        self.template = template
        #self.mat = self.build_mat()

    def get_rate(self, seq):
        bi, ei = self.input_range
        return np.exp(np.einsum('ij,ij', seq[bi:ei], self.mat))


class KineticModel():
    def __init__(self, model_params):
        """Kinetic model for a steady state enzymatic reaction.

        Parameters
        ----------
        param_file : str
            Yaml file that contains all parameters necessary to build kinetic model including sequence, state, rate, and output information.

        """
        # template : str, optional
        # A string sequence of DNA that all other sequences will be mutations
        # of. If None, all sequences will be randomly generated, by default
        # None
        self.model_params = model_params
        self.states = self.model_params['States']
        #self.template = self.model_params['Input'].get('template', None)
        self.template = None
        self.rates = [RateMat(rate, self.template)
                      for rate in self.model_params['Rates']]
        self.rate_names = [r.name for r in self.rates]

        (self.adj_mat,
         self.kinetic_mat,
         self.link_mat,
         self.links) = self.make_matrices()

        self.links.sort(key=lambda x: x.gid)
        self.build_ka_patterns()

    def make_matrices(self):
        """Make adjacency, kinetic, and link matrix to describe the kinetic
        reactions in model and implement the King-Altman method for finding the
        steady state occupancy of the state of the system.

        Returns
        -------
        numpy.ndarray
            [description]
        [[list]]
            [description]
        [[list]]
            [description]
        [[list]]
            [description]

        """
        adj_mat = np.zeros((len(self.states), len(self.states)))
        kin_mat = adj_mat.tolist()
        link_mat = adj_mat.tolist()
        links = []
        already_linked = []
        gid = 0
        for rate in self.rates:
            bs, es = rate.state_list  # beginning state and end state
            bsi, esi = self.states.index(bs), self.states.index(es)
            adj_mat[bsi, esi] = 1
            # Adjacency is non-directional in this mehtod
            adj_mat[esi, bsi] = 1
            # Save kinetic matrix for checking reactions later on
            kin_mat[bsi][esi] = rate

            # Created link matrix
            if rate.name not in already_linked:
                reverse_rate = False
                for rrate in self.rates:
                    if rrate.state_list[0] == es and rrate.state_list[1] == bs:
                        reverse_rate = rrate
                        break
                # Make a new link
                if not reverse_rate:
                    link_mat[bsi][esi] = Link([rate], (bs, es), gid)
                    link_mat[esi][bsi] = link_mat[bsi][esi]
                else:
                    link_mat[bsi][esi] = Link(
                        (rate, reverse_rate), (bs, es), gid)
                    link_mat[esi][bsi] = link_mat[bsi][esi]
                    already_linked += [reverse_rate]
                links += [link_mat[bsi][esi]]
                gid += 1

        return adj_mat, kin_mat, link_mat, links

    def build_ka_patterns(self):
        """Build a list of state sequences that are used in the King-Altman mehtod."""
        rows, cols = self.adj_mat.shape
        node_branch_list = []
        # Build up node branch list with first state cut from the link matrix
        for i in range(1, rows):
            node_branch_list += [[]]
            for j in range(cols):  # skip the first column
                if self.link_mat[i][j] != 0:
                    node_branch_list[-1] += [self.link_mat[i][j].gid]

        self.ka_patterns = wang_algebra_sequences(node_branch_list)

    def get_numerator(self, state):
        """ Get numerator of kinetic ratio related for state """
        rate_prod_list = []
        # Iterate through all patterns and choose the correct direction
        for kap in self.ka_patterns:
            # 1. Find all the links that have rates that end at the state
            # 2. If unused links still exist, find which ones have rates that end at the beginnings
            #    states of already chosen rates. Repeat process until all links are used
            # 2a. If no rates are found, return an empty list for that pattern
            rate_prod_list += [self.get_ka_rate_product([state], kap)]
        # Return a list of rate lists representing rate products removing any
        # empty lists
        return [r for r in rate_prod_list if r]

    def get_denominator(self):
        """ Denominator is just the sum of all the numerators"""
        return sum([self.get_numerator(state) for state in self.states], [])

    def get_ka_rate_product(self, end_states, link_ids):
        """Recursive function that finds the correct direction for links given a KA pattern sequence.
        If no path exists for that pattern because of irreversibility, an empty
        list is returned instead.
        """
        rate_list = []
        new_end_states = []
        used_lids = []
        for lid in link_ids:
            for es in end_states:
                for rate in self.links[lid].rates:
                    # If rate ends in an end state, add it to rate list
                    if rate.state_list[1] == es:
                        rate_list += [rate]
                        # Next recursive statement needs to know about new end
                        # state
                        new_end_states += [rate.state_list[0]]
                        used_lids += [lid]
                        break
                # Don't need to continue loop if link was used
                if lid in used_lids:
                    break
        # Find all links not used and pass them to next recursion step
        unused_lids = [lid for lid in link_ids if lid not in used_lids]
        if len(unused_lids) == len(link_ids):
            # Could not find kinetic rate for pattern
            return []
        if len(unused_lids) == 0:
            return rate_list
        next_rate = self.get_ka_rate_product(new_end_states, unused_lids)
        if not next_rate:
            return []
        return rate_list + next_rate

    def get_state_occupancy_func(self, state):
        """ Create and return a function that calculates the total occupancy of an enzymatic state given a sequence. """
        numer_list = self.get_numerator(state)
        denom_list = self.get_denominator()

        def get_state_occupancy(seq):
            numer = 0.
            denom = 0.
            for rlist in numer_list:
                term = 1.
                for nrate in rlist:
                    term *= nrate.get_rate(seq)
                numer += term

            for dlist in denom_list:
                term = 1.
                for drate in dlist:
                    term *= drate.get_rate(seq)
                denom += term

            return numer / denom

        return get_state_occupancy

    def get_ka_pattern_mat(self):
        """ Return a binary matrix that relates rate vector to a term vector
        t_i = A_{ij} k_j
         ^      ^ --|  ^------------|
     term vector   KA matrix     rate vector
  ( e.g [(k_1 * k_2 * k_3), ...)
        """
        denom_list = self.get_denominator()
        ka_mat = np.zeros((len(denom_list), len(self.rates)))
        for i, term in enumerate(denom_list):
            for r in term:
                ka_mat[i, self.rates.index(r)] = 1
        return ka_mat

    def gen_test_data(self, npoints=1000, contrib_rate_names=None,
                      pheno_map=None, mut_num=None, **kwargs):
        """Generate data in the form of a .csv to train a neural network to predict kinetics based off a sequence.

        Parameters
        ----------
        npoints : int, optional
            The number of data points to create, by default 1000
        contrib_rate_names : list, optional
            The rates from the model that contribute to the activity.
            Must not be empty. by default None
        pheno_map : str, optional
            The function used to change activity to an experimentally
            measurable phenotype, by default None
        mutation_num : int, optional
            The number of mutations to the template string sequence.
            If negative, all nucleotides will be changed except the
            negative number, by default -1
        """
        in_params = self.model_params['Input']
        assert(contrib_rate_names)
        if not self.template or mut_num is None:
            # Randomly generate array of sequences based off input parameters
            seq_arr = np.random.choice(in_params['values'],
                                       size=(npoints, in_params['seq_length']))
        else:
            assert(mut_num != 0)
            temp_seq = list(self.template)
            # Generate all sequences to
            seq_arr = np.repeat([temp_seq], npoints, axis=0)
            mut_num = mut_num if mut_num > 0 else len(temp_seq) - mut_num

            # Make a dictionary of lists with one of the label values removed
            # so that you can randomly choose from the correct list later when
            # making mutations
            opt_dict = {}
            for key in in_params['values']:
                opt_dict[key] = [v for v in in_params['values'] if v != key]

            # Choose all indices to mutate for each sequence
            # ind_choice = np.random.choice(np.arange(len(temp_seq)),
            #                               size=(npoints, mut_num),
            #                               replace=False)
            for i in range(npoints):
                ind_choice = np.random.choice(len(temp_seq),
                                              mut_num,
                                              replace=False)
                for mut_ind in ind_choice:
                    mut_label = seq_arr[i, mut_ind]
                    seq_arr[i, mut_ind] = np.random.choice(opt_dict[mut_label])

        lab_enc, one_enc = make_encoders(in_params['values'])
        # Create occupancy functions for states that contribute to pheno_map
        # based on the list of contrib_rates
        contrib_rates = []
        occupancy_funcs = []
        for name in contrib_rate_names:
            for rate in self.rates:
                if rate.name == name:
                    contrib_rates += [rate]
                    # Get the pre-transition state of the contributing rate
                    occupancy_funcs += [
                        self.get_state_occupancy_func(rate.state_list[0])]

        act_arr = np.zeros((npoints))
        pheno_arr = np.zeros((npoints))
        pheno_map_func = eval(pheno_map) if pheno_map else lambda x: x

        for i, seq in enumerate(seq_arr):
            tmp = lab_enc.transform(seq)
            # one hot encode random sequence i keeping original length
            seq_ohe = one_enc.transform(tmp.reshape(-1, 1))
            # loop over occupancy functions and associated contrib_rates
            for rate, ofunc in zip(contrib_rates, occupancy_funcs):
                # Get occupancy and multiply by contrib_rate value
                # Sum all terms to get pheno_map
                act_arr[i] += rate.get_rate(seq_ohe) * ofunc(seq_ohe)
            pheno_arr[i] = pheno_map_func(act_arr[i])

        # Save phenotype to file
        comb_arr = np.hstack(
            (seq_arr, act_arr.reshape(-1, 1), pheno_arr.reshape(-1, 1)))
        df = pd.DataFrame(comb_arr)
        file_name = Path(self.save_str + ('.csv'))
        df.to_csv(file_name)
        self.save_ka_matrix()
        self.save_rate_contrib_matrix(contrib_rate_names)

    def save_ka_matrix(self):
        ka_mat = self.get_ka_pattern_mat()
        np.savetxt(Path(self.save_str + '_ka_mat.nptxt'), ka_mat)

    def save_rate_contrib_matrix(self, contrib_rate_names, save=True):
        denom_list = self.get_denominator()
        contrib_mat = np.zeros((len(self.rates), len(denom_list)))
        for i, crate in enumerate(self.rates):
            if crate.name in contrib_rate_names:
                start_state = crate.state_list[0]
                nterms = self.get_numerator(start_state)
                for j, kap in enumerate(denom_list):
                    if kap in nterms:
                        contrib_mat[i, j] = 1.
        if save is True:
            np.savetxt(Path(self.save_str + '_rate_contrib_mat.nptxt'),
                       contrib_mat)
        return contrib_mat

    def get_rate_contrib_matrix(self):
        #return np.loadtxt(
        #    Path(self.save_str + '_rate_contrib_mat.nptxt'))
        contrib_rate_names = self.model_params['Data']['contrib_rate_names']
        return self.save_rate_contrib_matrix(contrib_rate_names, save=False)


class Link():
    def __init__(self, rates, states, gid):
        self.rates = rates
        self.states = states
        self.gid = gid


