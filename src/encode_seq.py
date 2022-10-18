# -*- coding: utf-8 -*-

import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
import warnings


class Encoder_CRISPR_Net:
    def __init__(self, on_seq, off_seq, with_category = False, label = None, with_reg_val = False, value = None):
        self.tlen = 25
        self.on_seq = "-" *(self.tlen-len(on_seq)) +  on_seq
        self.off_seq = "-" *(self.tlen-len(off_seq)) + off_seq
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        if with_category:
            self.label = label
        if with_reg_val:
            self.value = value
        #self.encode_on_off()
    
    def encode_on_off(self):
        return self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)
        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_on_off_dim7(self):
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)


def get_letter_index(build_indel=True):
    # pre-defined letter index
    # match : 4 letters, 0-3
    ltidx = {(x,x):i for i,x in enumerate('ACGT')}
    # substitution : x->y, 4-7
    ltidx.update({(x,y):(ltidx[(x,x)], i+4) for x in 'ACGT' for i, y in enumerate('ACGT') if y!=x })
    if build_indel:
        # insertion : NA->y, 8-11
        ltidx.update({('-', x):i+8 for i,x in enumerate('ACGT')})
        ltidx.update({('_', x):i+8 for i,x in enumerate('ACGT')})
        # deletion : x->NA, 12
        ltidx.update({(x,'-'):(ltidx[(x,x)], 12) for i,x in enumerate('ACGT')})
        ltidx.update({(x,'_'):(ltidx[(x,x)], 12) for i,x in enumerate('ACGT')})
    return ltidx


def make_alignment(ref, alt, maxlen=25):
    alt = Seq(alt)
    alt = alt[::-1]
    ref = ref[::-1]
    # m: A match score is the score of identical chars, otherwise mismatch score
    # d: The sequences have different open and extend gap penalties.
    aln = pairwise2.align.localxd(ref, alt, -1, -0.1, -1, 0)
    if len(aln[0][0]) > maxlen: # increase gap open penalty to avoid too many gaps
        aln = pairwise2.align.localxd(ref, alt, -5, -0.1, -5, 0)
        if len(aln[0][0]) > maxlen:
            aln = [(ref, alt)]
    return aln[0]


def featurize_alignment(alignments, ltidx, build_indel=True, include_ref=False, maxlen=25, verbose=False):
    mats = []
    for j, aln in enumerate(alignments):
        if build_indel:
            fea = np.zeros((maxlen, 13))
        else:
            fea = np.zeros((maxlen, 8))
        assert len(aln[0]) <= maxlen, "alignment {} larger than maxlen: {}".format(j, aln)
        if build_indel is False and ('-' in aln[0] or '-' in aln[1]):
            mats.append(None)
        else:
            p = 0
            for i in range(len(aln[0])):
                k = (aln[0][i], aln[1][i])
                if not k in ltidx:
                    if verbose:
                        warnings.warn("found alignment not in letter, sample %i" %j)
                    p += 1
                    continue
                #fea[p, ltidx[k]] = 1
                #p += 1
                if k[0]=="-" or k[1]=="-": # is indel
                    #if build_indel:
                    #    fea[p, ltidx[k]] = 1
                    #    p += 1
                    if k[1]=="-": # target has gap - deletion
                        fea[p, ltidx[k]] = 1
                        p += 1
                    else:           # gRNA has gap - insertion
                        fea[max(0,p-1), ltidx[k]] = 1
                else:
                    fea[p, ltidx[k]] = 1
                    p += 1
                if verbose: print(k, p)
            mats.append(fea)
    mats = np.array(mats)
    if include_ref is False:
        mats = mats[:, :, 4:]
    return mats


class Encoder_Electrum(Encoder_CRISPR_Net):
    def __init__(self, on_seq, off_seq, build_indel=True, include_ref=True, is_aligned=True, with_category=False, label=None, with_reg_val=False, value=None):
        super().__init__(on_seq, off_seq, with_category, label, with_reg_val, value)
        self.is_aligned = is_aligned
        self.build_indel = build_indel
        self.include_ref = include_ref
        self.encoded_dict_indel = get_letter_index(build_indel=self.build_indel)
        # reverse the seq for Electrum
        self.on_seq = self.on_seq[::-1]
        self.off_seq = self.off_seq[::-1]
    
    def encode_on_off(self):
        if self.is_aligned is False:
            aln = make_alignment(ref=self.on_seq, alt=self.off_seq, maxlen=self.tlen)
        else:
            aln = [self.on_seq, self.off_seq]
        fea = self.featurize_alignment(aln=aln)
        self.on_off_code = fea

    def featurize_alignment(self, aln, verbose=False):
        if self.build_indel:
            fea = np.zeros((self.tlen, 13))
        else:
            fea = np.zeros((self.tlen, 8))
        assert len(aln[0]) <= self.tlen, "alignment {} larger than maxlen: {}".format(j, aln)
        if self.build_indel is False and ('-' in aln[0] or '-' in aln[1]):
            return None
        else:
            p = 0
            for i in range(len(aln[0])):
                k = (aln[0][i], aln[1][i])
                if not k in self.encoded_dict_indel:
                    if verbose:
                        warnings.warn("found alignment not in letter, sample %i" %j)
                    p += 1
                    continue
                if k[0]=="-" or k[1]=="-": # is indel
                    if k[1]=="-": # target has gap - deletion
                        fea[p, self.encoded_dict_indel[k]] = 1
                        p += 1
                    else:           # gRNA has gap - insertion
                        fea[max(0,p-1), self.encoded_dict_indel[k]] = 1
                else:
                    fea[p, self.encoded_dict_indel[k]] = 1
                    p += 1
                if verbose: print(k, p)
            if self.include_ref is False:
                fea = fea[:, 4:]
            return fea


# Alias
Encoder = Encoder_Electrum

# Testing
# e = Encoder(on_seq="AGCTGA", off_seq="CG_GTT")
# print(e.on_off_code)












