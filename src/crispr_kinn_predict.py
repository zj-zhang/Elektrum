"""a script that wraps up the calling and parsing of CRISPR-OffTarget predictions

FZZ, 2022.03.15
"""

import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Seq import Seq
import warnings
from src.reload import reload_from_dir
from src.neural_network_builder import KineticNeuralNetworkBuilder, KineticEigenModelBuilder
#import tensorflow as tf
from amber.utils import corrected_tf as tf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score
import scipy.stats as ss


config = {
    'kinn_1': "outputs/bak_20220515/KINN-wtCas9_cleave_rate_log-finkelstein-0-rep2-gRNA1",
    'kinn_2': "outputs/bak_20220515/KINN-wtCas9_cleave_rate_log-finkelstein-0-rep1-gRNA2",
    'dcnn_1': "outputs/CNN-wtCas9_cleave_rate_log-0/",
    'dcnn_2': "outputs/CNN-wtCas9_cleave_rate_log-1/",
}

# trainEnv parameters
evo_params = dict(
    model_fn = KineticNeuralNetworkBuilder,
    #model_fn = KineticEigenModelBuilder,
    samps_per_gen = 10,   # how many arcs to sample in each generation; important
    max_gen = 200,
    patience = 50,
    n_warmup_gen = 0,
    #train_data = (x_train, y_train),
    #test_data = (x_test, y_test)
)

# manager configs
manager_kwargs={
    'output_op': lambda: tf.keras.layers.Lambda(lambda x: tf.math.log(tf.clip_by_value(x, 10**-5, 10**-1))/np.log(10), name="output_log"),  # change the clip as well
    #'output_op': lambda: tf.keras.layers.Lambda(lambda x: tf.math.log(tf.math.maximum(tf.reshape(- x[:,1], (-1,1)), 10**-5))/np.log(10), name="output_slice"),
    'n_feats': 25,  # remember to change this!!
    'n_channels': 9,
    'batch_size': 128,
    'epochs': 30,
    'earlystop': 10,
    'verbose': 0
}

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


def plot_dataframe(data):
    fig, axs = plt.subplots(1,2, figsize=(13,6))
    plot_df = data.query('Read>0')
    ax = axs[1]
    sns.scatterplot(x='kinn', y=np.log10(plot_df.Read+1), hue='sgRNA_type', alpha=0.5, data=plot_df, ax=ax)
    ax.set_ylabel('log10(Read+1)')
    ax.set_xlabel('KINN log10(cleavage rate)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10})
    pcc = ss.pearsonr(plot_df['kinn'], np.log10(plot_df['Read']+1))
    ax.set_title("Positive Set Predictions (OffTargets)\nPearson=%.3f, p=%.3f, n=%i" % (pcc[0], pcc[1], plot_df.shape[0]))
    #ax = axs[1,1]
    #sns.scatterplot(x='dcnn', y=np.log10(plot_df.Read+1), hue='sgRNA_type', alpha=0.5, data=plot_df, ax=ax)
    #ax.set_ylabel('log10(Read+1)')
    #ax.set_xlabel('DCNN log10(cleavage rate)')
    #ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 10})
    #pcc = ss.pearsonr(plot_df['dcnn'], np.log10(plot_df['Read']+1))
    #ax.set_title("Positive Set Predictions (OffTargets)\nPearson=%.3f, p=%.3f, n=%i" % (pcc[0], pcc[1], plot_df.shape[0]))


    plot_df = data.query('Read==0')
    q = np.arange(0,100.01,0.01)
    ax = axs[0]
    sns.lineplot(x=q, y=np.percentile(plot_df.kinn, q), marker="o", markersize=6, ax=ax)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("KINN log10(cleavage rate)")
    ax.set_title("Negative Set Predictions (non-edited)\nn=%i"%(plot_df.shape[0]))
    #ax = axs[1,0]
    #sns.lineplot(x=q, y=np.percentile(plot_df.dcnn, q), marker="o", markersize=6, ax=ax)
    #ax.set_xlabel("Percentile")
    #ax.set_ylabel("DCNN log10(cleavage rate)")
    #ax.set_title("Negative Set Predictions (non-edited)\nn=%i"%(plot_df.shape[0]))

    fig.tight_layout()
    return fig


def predict_on_dataframe(data, is_aligned=True):
    """Assumes the first and second columns are sgRNA,OffTarget respectively
    """
    ltidx = get_letter_index(build_indel=True)
    if is_aligned:
        alignments = [x[1].str[::-1].tolist() for x in tqdm(data.iloc[:,[0,1]].iterrows(), total=data.shape[0])]
    else:
        # sanitize sequences and perform alignments
        raise NotImplementedError
    fea = featurize_alignment(alignments, ltidx)
    # load kinn
    sess = tf.Session()
    kinn_1 = reload_from_dir(wd=config['kinn_1'],
                             sess=sess,
                             manager_kwargs=manager_kwargs,
                             model_fn=evo_params['model_fn']
                             )
    kinn_2 = reload_from_dir(wd=config['kinn_2'],
                             sess=sess,
                             manager_kwargs=manager_kwargs,
                             model_fn=evo_params['model_fn']
                             )
    # load cnn
    wd = config['dcnn_1']
    train_hist = pd.read_table(os.path.join(wd, "train_history.csv"), sep=",", header=None)
    best_trial_id = train_hist.sort_values(2, ascending=False).head(1)[0]
    dcnn_1 = tf.keras.models.load_model(os.path.join(wd, "weights", "trial_%i"%best_trial_id, "bestmodel.h5"))

    wd = config['dcnn_2']
    train_hist = pd.read_table(os.path.join(wd, "train_history.csv"), sep=",", header=None)
    best_trial_id = train_hist.sort_values(2, ascending=False).head(1)[0]
    dcnn_2 = tf.keras.models.load_model(os.path.join(wd, "weights", "trial_%i"%best_trial_id, "bestmodel.h5"))

    data['kinn_1'] = kinn_1.predict(fea)
    data['kinn_2'] = kinn_2.predict(fea)
    data['dcnn_1'] = dcnn_1.predict(fea)
    data['dcnn_2'] = dcnn_2.predict(fea)
    data['kinn'] = data[['kinn_1', 'kinn_2']].mean(axis=1)
    data['dcnn'] = data[['dcnn_1', 'dcnn_2']].mean(axis=1)

    metrics = {
            'auroc.kinn': roc_auc_score(y_true=data.label, y_score=data.kinn),
            'aupr.kinn': average_precision_score(y_true=data.label, y_score=data.kinn),
            'auroc.cnn': roc_auc_score(y_true=data.label, y_score=data.dcnn),
            'aupr.cnn': average_precision_score(y_true=data.label, y_score=data.dcnn),
            'num.total': data.shape[0],
            'num.unique_gRNAs': len(set(data['sgRNA_type'])),
            'num.offtarget': data.label.sum()
    }
    return data, fea, metrics
