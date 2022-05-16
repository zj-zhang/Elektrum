import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.crispr_kinn_predict import featurize_alignment, get_letter_index


def get_sim_ness_data(seed=111):
    ltidx = get_letter_index()
    df = pd.read_table("/mnt/home/alamson/ceph/DATA/CRISPR/KineticSims/22-05-12_cas9_kinn_deplete/cas9_kinn_deplete_full_data.tsv")
    df = df.query('rate_fit != 1.0')
    df['rate_fit'] = np.clip(df['rate_fit'], 1e-8, 10)
    #ref = "TCGGTAGGATCGTAAGATAGTATTCAGGACCCCGTTAACCATTTCGAAAG"
    ref = df.iloc[0]['seq']
    alignments = [(ref, r['seq']) for _, r in df.iterrows() ]
    x = featurize_alignment(alignments=alignments, ltidx=ltidx, maxlen=50)
    y = np.log(df['rate_fit'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    return (x_train, y_train), (x_test, y_test)


def load_finkelstein_data(target='wtCas9_cleave_rate_log', make_switch=False):
    x = np.load('./data/compiled_X_1.npy')
    y = np.load('./data/compiled_Y_1.npy')
    y = np.log(10**y)

    x_2 = np.load('./data/compiled_X_2.npy')
    y_2 = np.load('./data/compiled_Y_2.npy')
    y_2 = np.log(10**y_2)
    #y_2 = 10**y_2

    if make_switch is True:
        x, y, x_2, y_2 = x_2, y_2, x, y
    with open('./data/y_col_annot.txt', 'r') as f:
        label_annot = [x.strip() for x in f]
        label_annot = {x:i for i,x in enumerate(label_annot)}
    # for training data, do NOT split since the MPKA is non-redundent
    x_train, x_test, y_train, y_test = x, None, y, None
    x2_train, x2_test, y2_train, y2_test = train_test_split(x_2, y_2, test_size=0.2, random_state=888)
    tar_to_train = label_annot[target]
    #return x_train, y_train[:, tar_to_train], None, None, \
    #    x2_train, y2_train[:, tar_to_train], x2_test, y2_test[:, tar_to_train]
    return (x_train, y_train[:, tar_to_train]), (x2_train, y2_train[:, tar_to_train])


