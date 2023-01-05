import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.crispr_kinn_predict import featurize_alignment, get_letter_index


def get_sim_deplete_data(fp, seed=111, logbase=None):
    ltidx = get_letter_index()
    df = pd.read_table(fp)
    #df = df.query('rate_fit != 1.0')
    print(df.shape)
    df = df.query('rate_fit_rsqr > 0.98')
    print(df.shape)
    print(df['rate_fit_rsqr'].min(), df['rate_fit_rsqr'].max())
    ref = df.iloc[0]['seq']
    alignments = [(ref, r['seq']) for _, r in df.iterrows()]
    x = featurize_alignment(alignments=alignments, ltidx=ltidx, maxlen=50)
    y = df['rate_fit']
    #y = - df['first_eigval']
    if logbase:
        y = np.log(y) / np.log(logbase)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed)
    return (x_train, y_train), (x_test, y_test)


def load_finkelstein_data(target='wtCas9_cleave_rate_log', 
                          make_switch=False, logbase=None, 
                          include_ref=False, return_remainder=False):
    x = np.load('./test_files/test_data_transfer/compiled_X_1.npy')
    if include_ref is False:
        x = x[:, :, 4:]
    y = np.load('./test_files/test_data_transfer/compiled_Y_1.npy')
    y = 10**y
    if logbase:
        y = np.log(y) / np.log(logbase)

    x_2 = np.load('./test_files/test_data_transfer/compiled_X_2.npy')
    if include_ref is False:
        x_2 = x_2[:, :, 4:]
    y_2 = np.load('./test_files/test_data_transfer/compiled_Y_2.npy')
    y_2 = 10**y_2
    if logbase:
        y_2 = np.log(y_2) / np.log(logbase)

    if make_switch is True:
        x, y, x_2, y_2 = x_2, y_2, x, y
    with open('./test_files/test_data_transfer/y_col_annot.txt', 'r') as f:
        label_annot = [x.strip() for x in f]
        label_annot = {x: i for i, x in enumerate(label_annot)}
    # for training data, do NOT split since the MPKA is non-redundent
    x_train, x_test, y_train, y_test = x, None, y, None
    x2_train, x2_test, y2_train, y2_test = train_test_split(
        x_2, y_2, test_size=0.2, random_state=888)
    tar_to_train = label_annot[target]
    if return_remainder is False:
        return (x_train, y_train[:, tar_to_train]), (x2_train, y2_train[:, tar_to_train])
    else:
        return x2_test, y2_test[:, tar_to_train]
