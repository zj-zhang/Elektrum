#!/usr/bin/env python
# coding: utf-8

# # Probablistic model building genetic algorithm


from src.kinetic_model import KineticModel, modelSpace_to_modelParams
from src.neural_network_builder import KineticNeuralNetworkBuilder

import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import scipy.stats as ss
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
import shutil
import gc
import argparse
import pickle
import amber
print(amber.__version__)
from amber.architect import pmbga
from amber.architect import ModelSpace, Operation


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, choices="""wtCas9_cleave_rate_log
Cas9_enh_cleave_rate_log
Cas9_hypa_cleave_rate_log
Cas9_HF1_cleave_rate_log
wtCas9_cleave_rate_log_specificity
Cas9_enh_cleave_rate_log_specificity
Cas9_hypa_cleave_rate_log_specificity
Cas9_HF1_cleave_rate_log_specificity
wtCas9_ndABA
Cas9_enh_ndABA
Cas9_hypa_ndABA
Cas9_HF1_ndABA""".split(), required=True)
    parser.add_argument('--ms', type=str, choices=['finkelstein', 'uniform'], required=True)
    parser.add_argument('--wd', type=str, required=True)
    parser.add_argument('--n-states', type=int, default=4, required=False)
    parser.add_argument('--win-size', type=int, default=7, required=False)

    args = parser.parse_args()
    os.makedirs(args.wd, exist_ok=True)
    return args


def run():
    args = parse()
    if args.ms == "finkelstein":
        kinn_model_space = get_finkelstein_ms()
    else:
        kinn_model_space = get_uniform_ms(n_states=args.n_states, st_win_size=args.win_size)

    print(kinn_model_space)
    controller = pmbga.ProbaModelBuildGeneticAlgo(
                model_space=kinn_model_space,
                buffer_type='population',
                buffer_size=50,  # buffer size controlls the max history going back
                batch_size=1,   # batch size does not matter in this case; all arcs will be retrieved
            )
    x_train, x_test, y_train, y_test = load_data(target=args.target)
    # trainEnv parameters
    samps_per_gen = 10   # how many arcs to sample in each generation; important
    max_gen = 2000
    epsilon = 0.05
    patience = 500
    n_warmup_gen = -1

    # get prior probas
    #_, old_probs = compute_eps(controller.model_space_probs)

    # ## A fancy For-Loop that does the work for `amber.architect.trainEnv`
    hist = []
    pc_cnt = 0
    best_indv = 0
    stat_df = pd.DataFrame(columns=['Generation', 'GenAvg', 'Best', 'PostVar'])
    for generation in range(max_gen):
        try:
            start = time.time()
            has_impr = False
            #for _ in tqdm(range(samps_per_gen), total=samps_per_gen, position=0, leave=True):
            for _ in range(samps_per_gen):
                # get arc
                arc, _ = controller.get_action()
                # get reward
                try:
                    test_reward = get_reward_pipeline(arc, 
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            wd=args.wd
                            )
                #except ValueError:
                #    test_reward = 0
                except Exception as e:
                    raise e
                rate_df = None
                # update best, or increase patience counter
                if test_reward > best_indv:
                    best_indv = test_reward
                    has_impr = True
                    shutil.move(os.path.join(args.wd, "bestmodel.h5"), os.path.join(args.wd, "AmberSearchBestModel.h5"))
                    shutil.move(os.path.join(args.wd, "model_params.pkl"), os.path.join(args.wd, "AmberSearchBestModel_config.pkl"))
                    
                # store
                _ = controller.store(action=arc, reward=test_reward)
                hist.append({'gen': generation, 'arc':arc, 'test_reward': test_reward, 'rate_df': rate_df})
            end = time.time()
            if generation < n_warmup_gen:
                print(f"Gen {generation} < {n_warmup_gen} warmup.. skipped - Time %.2f" % (end-start), flush=True)
                continue
            _ = controller.train(episode=generation, working_dir=".")
            #delta, old_probs = compute_eps(controller.model_space_probs, old_probs)
            delta = 0
            post_vars = [np.var(x.sample(size=100)) for _, x in controller.model_space_probs.items()]
            stat_df = stat_df.append({
                'Generation': generation,
                'GenAvg': controller.buffer.r_bias,
                'Best': best_indv,
                'PostVar': np.mean(post_vars)
            }, ignore_index=True)
            print("[%s] Gen %i - Mean fitness %.3f - Best %.4f - PostVar %.3f - Eps %.3f - Time %.2f" % (
                datetime.now().strftime("%H:%M:%S"),
                generation, 
                controller.buffer.r_bias, 
                best_indv, 
                np.mean(post_vars),
                delta,
                end-start), flush=True)
            #if delta < epsilon:
            #    print("stop due to convergence criteria")
            #    break
            pc_cnt = 0 if has_impr else pc_cnt+1
            if pc_cnt >= patience:
                print("early-stop due to max patience w/o improvement")
                break
        except KeyboardInterrupt:
            print("user interrupted")
            break

    # write out
    a = pd.DataFrame(hist)
    a['arc'] = ['|'.join([f"{x.Layer_attributes['RANGE_ST']}-{x.Layer_attributes['RANGE_ST']+x.Layer_attributes['RANGE_D']}-k{x.Layer_attributes['kernel_size']}" for x in entry]) for entry in a['arc']]
    a.drop(columns=['rate_df'], inplace=True)
    a.to_csv(os.path.join(args.wd,"train_history.tsv"), sep="\t", index=False)
    ax = stat_df.plot.line(x='Generation', y=['GenAvg', 'Best'])
    ax.set_ylabel("Reward (Pearson correlation)")
    ax.set_xlabel("Generation")
    plt.savefig(os.path.join(args.wd, "reward_vs_time.png"))

    # plot
    make_plots(controller, canvas_nrow=np.ceil(np.sqrt(len(kinn_model_space))), wd=args.wd)
    return controller


def load_data(target):
    x = np.load('./data/compiled_X.npy')
    y = np.load('./data/compiled_Y.npy')
    with open('./data/y_col_annot.txt', 'r') as f:
        label_annot = [x.strip() for x in f]
        label_annot = {x:i for i,x in enumerate(label_annot)}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777)
    target_idx = label_annot[target]
    y_train = y_train[:, target_idx]
    y_test = y_test[:, target_idx]
    return x_train, x_test, y_train, y_test


# Model Space
def get_finkelstein_ms():
    """model space based on https://www.biorxiv.org/content/10.1101/2020.05.21.108613v2
    """
    ks_choices=[1,3,5]
    kinn_model_space = ModelSpace.from_dict([
        # k_on, sol -> open R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='0', TARGET='1',
              kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=1),
              EDGE=1,
              RANGE_ST=0,
              RANGE_D=3,
         )],
        # k_off, open R-loop -> sol
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='0', 
              kernel_size=pmbga.Categorical(choices=[1,3,5], prior_cnt=1),
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[0,1,2], prior_cnt=1),
              RANGE_D=3
         )],
        # k_OI, open R-loop -> intermediate R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='2', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=1), 
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[3,4,5,6,7,8,9,10,11,12], 
                  prior_cnt=1),
              RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
         )],
        # k_IO, intermediate R-loop -> open R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='1', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=1), 
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[3,4,5,6,7,8,9,10,11,12],
                  prior_cnt=1),
              RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),             
         )],
        # k_IC, intermediate R-loop -> closed R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='3', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=[1]*3), 
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[11,12,13,14,15,16,17,18,19], prior_cnt=1),
              RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),     
         )],
        # k_CI, closed R-loop -> intermediate R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='2', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=[1]*3),
              EDGE=1,        
              RANGE_ST=pmbga.Categorical(choices=[11,12,13,14,15,16,17,18,19], prior_cnt=1),
              RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),          
         )],
        # k_30
        [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='0', 
              kernel_size=pmbga.Categorical(choices=[1,3,5,7], prior_cnt=1),
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=np.arange(0,19), prior_cnt=1),
              RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
              CONTRIB=1
         )],
    ])
    return kinn_model_space


def get_uniform_ms(n_states, st_win_size=7):
    """an evenly-spaced model space, separating 20nt for given n_states
    """
    st_win = np.arange(st_win_size) - st_win_size//2
    anchors = {s:i for s,i in enumerate(np.arange(0, 23, np.ceil(23/n_states), dtype='int'))}
    ls = []
    default_ks = lambda: pmbga.Categorical(choices=[1,3,5], prior_cnt=1)
    default_d = lambda: pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1)
    default_st = lambda a: pmbga.Categorical(choices=np.clip(a+st_win, 0, 23), prior_cnt=1)
    for s in range(0, n_states-1):
        ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s), TARGET=str(s+1), EDGE=1,
            kernel_size=default_ks(),
            RANGE_ST=default_st(anchors[s]),
            RANGE_D=default_d()
            )])
        ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s+1), TARGET=str(s), EDGE=1,
            kernel_size=default_ks(),
            RANGE_ST=default_st(anchors[s]),
            RANGE_D=default_d()
            )])
    # last rate: cleavage, irreversible
    ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s+1), TARGET='0', EDGE=1,
            kernel_size=pmbga.Categorical(choices=[1,2,3,4,5,6], prior_cnt=[1]*6),
            RANGE_ST=pmbga.Categorical(choices=np.arange(0, 20), prior_cnt=[1]*20),
            RANGE_D=default_d(),
            CONTRIB=1
            )])
    return ModelSpace.from_dict(ls)


# ## Components before they are implemented in AMBER

## NEEDS RE-WORK
# poorman's manager get reward
def get_reward_pipeline(model_arcs, x_train, y_train, x_test, y_test, wd):
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)
    model_params = modelSpace_to_modelParams(model_arcs)
    pickle.dump(model_params, open(os.path.join(wd, "model_params.pkl"), "wb"))
    tf.reset_default_graph()
    with train_graph.as_default(), train_sess.as_default():
        kinn_test = KineticModel(model_params)
        mb = KineticNeuralNetworkBuilder(kinn=kinn_test, session=train_sess, n_channels=13,
                replace_conv_by_fc=True)
        # train and test
        mb.build(optimizer='adam', plot=False, output_act=False)
        model = mb.model
        x_train_b = mb.blockify_seq_ohe(x_train)
        x_test_b = mb.blockify_seq_ohe(x_test)
        checkpointer = ModelCheckpoint(filepath=os.path.join(wd,"bestmodel.h5"), mode='min', verbose=0, save_best_only=True,
                               save_weights_only=True)
        earlystopper = EarlyStopping(
            monitor="val_loss",
            mode='min',
            patience=5,
            verbose=0)

        hist = model.fit(x_train_b, y_train,
                  batch_size=768,
                  validation_split=0.1,
                  callbacks=[checkpointer, earlystopper],
                  epochs=75, verbose=0)
        model.load_weights(os.path.join(wd,"bestmodel.h5"))
        y_hat = model.predict(x_test_b).flatten()
        test_reward = ss.pearsonr(y_hat, y_test)[0]
    del train_graph, train_sess
    del model, hist
    tf.keras.backend.clear_session() # THIS IS IMPORTANT!!!
    gc.collect()
    return test_reward




def compute_eps(model_space_probs, old_probs=None):
    delta = []
    samp_probs = {}
    for p in model_space_probs:
        #print(p)
        samp_probs[p] = model_space_probs[p].sample(size=10000)
        n = np.percentile(samp_probs[p], [10, 20, 30, 40, 50, 60, 70, 80, 90])
        if old_probs is None:
            delta.append( np.mean(np.abs(n)) )
        else:
            o = np.percentile(old_probs[p], [10, 20, 30, 40, 50, 60, 70, 80, 90])
            delta.append( np.mean(np.abs(o - n)) )
    return np.mean(delta), samp_probs 



def make_plots(controller, canvas_nrow, wd):
    canvas_nrow = int(canvas_nrow)
    # START SITE
    fig, axs_ = plt.subplots(canvas_nrow, canvas_nrow, figsize=(4.5*canvas_nrow,4.5*canvas_nrow))
    axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
    for k in controller.model_space_probs:
        if k[-1] == 'RANGE_ST':
            try:
                d = controller.model_space_probs[k].sample(size=1000)
            except:
                continue
            ax = axs[k[0]]
            sns.distplot(d, label="Post", ax=ax)
            sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
            ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))

    fig.tight_layout()
    fig.savefig(os.path.join(wd,"range_st.png"))

    # CONV RANGE
    fig, axs_ = plt.subplots(canvas_nrow, canvas_nrow, figsize=(4.5*canvas_nrow,4.5*canvas_nrow))
    axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
    for k in controller.model_space_probs:
        if k[-1] == 'RANGE_D':
            d = controller.model_space_probs[k].sample(size=1000)
            ax = axs[k[0]]
            sns.distplot(d, ax=ax)
            sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
            ax.set_title(
                    ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
    fig.tight_layout()
    fig.savefig(os.path.join(wd,"range_d.png"))

    # KERNEL SIZE 
    fig, axs_ = plt.subplots(canvas_nrow, canvas_nrow, figsize=(4.5*canvas_nrow,4.5*canvas_nrow))
    axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
    for k in controller.model_space_probs:
        if k[-1] == 'kernel_size':
            d = controller.model_space_probs[k].sample(size=1000)
            ax = axs[k[0]]
            sns.distplot(d, ax=ax)
            sns.distplot(controller.model_space_probs[k].prior_dist, ax=ax)
            ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
    fig.tight_layout()
    fig.savefig(os.path.join(wd,"kernel_size.png"))



if __name__ == "__main__":
    if not amber.utils.run_from_ipython():
        run()

