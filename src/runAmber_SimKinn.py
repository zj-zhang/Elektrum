#!/usr/bin/env python
# coding: utf-8

"""
Example usage:
python src/runAmber_SimKinn.py --wd outputs/test --data-file data/sim_data/21-11-1_test_1/test_1.csv --param-file data/sim_data/21-11-1_test_1/test_1_model_params.yaml
"""


# # Probablistic model building genetic algorithm

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from src.kinetic_model import KineticModel, modelSpace_to_modelParams
from src.neural_network_builder import KineticNeuralNetworkBuilder

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats as ss
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse

import amber
print(amber.__version__)
from amber.architect import pmbga
from amber.architect import ModelSpace, Operation


parser = argparse.ArgumentParser()
parser.add_argument('--disable-posterior', action="store_true", default=False)
parser.add_argument('--patience', type=int, default=200, required=False)
parser.add_argument('--max-gen', type=int, default=600, required=False)
parser.add_argument('--samps-per-gen', type=int, default=5, required=False)
parser.add_argument('--wd', type=str, required=True)
parser.add_argument('--data-file', type=str, required=True)
parser.add_argument('--param-file', type=str, required=True)



args = parser.parse_args()

# trainEnv parameters
samps_per_gen = args.samps_per_gen   # how many arcs to sample in each generation; important
max_gen = args.max_gen
patience = args.patience
n_warmup_gen = -1
disable_posterior_update = args.disable_posterior
wd = args.wd

os.makedirs(wd, exist_ok=True)

# In[2]:


kinn_sty = {
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "lines.linewidth": 1,
    "lines.markersize": 10,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "font.size": 18,
    "font.sans-serif": 'Helvetica',
    "text.usetex": False,
    'mathtext.fontset': 'cm',
}
plt.style.use(kinn_sty)

# ground-truth params for synthetic data
with open(args.param_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    kinn_gr = KineticModel(config)


def kinn_train_fn(model, traindata, epochs, plot=False, verbose=0):
    """this can use lots of fine-tuning arguments to parse; for the purpose of simulated data, keep it as is

    Parameters
    ----------
    epochs : int
        total epochs to train
    plot : bool
        save history plot
    """
    model_fp = os.path.join(args.wd, "bestmodel.h5")
    checkpointer = ModelCheckpoint(filepath=model_fp, mode='min', verbose=verbose, save_best_only=True,
                                   save_weights_only=True)
    earlystopper = EarlyStopping(
        monitor="val_loss",
        mode='min',
        patience=20,
        verbose=0)
    history = model.fit(traindata[0], traindata[1],
                         batch_size=512,
                         callbacks=[checkpointer, earlystopper],
                         validation_split=0.2, epochs=epochs, verbose=0)
    model.load_weights(model_fp)
    if plot:
        plt.figure()
        plt.plot(history.history['loss'], color='blue')
        plt.plot(history.history['val_loss'], color='orange')
        plt.title('Model loss', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.xlabel('epoch', fontsize=12)
        plt.legend(['train', 'validation'])
        plt.savefig(os.path.join(args.wd, 'hist.png'))
    return model


def kinn_test_fn(model, testdata, plot=False):
    y_pred = model.predict(testdata[0])
    pcc = ss.pearsonr(y_pred.flatten(), testdata[1])
    #print(pcc)
    if plot:
        plt.clf()
        plt.scatter(x=y_pred.flatten(), y=self.testdata[1])
        ax = plt.gca()
        ax.set_xlabel('pred')
        ax.set_ylabel('obs')
        ax.set_title('pearson=%.3f' % pcc[0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(args.wd, 'test.png'))
    return pcc


def kinn_interpret_fn(mb):
    """match each learned rate parameters to the ground truth stored in biophysics model
    """
    layer_dict = {l.name:l for l in mb.model.layers}
    model_weights = {
        name: layer_dict[name].get_weights().copy() for name in layer_dict}
    rate_params = {}
    for k in range(len(mb.kinn.rates)):
        rate_params['rate_k%i' % k] = {
            'estimate': np.squeeze(model_weights['conv_k%i' % k][0]),
            'truth': kinn_gr.rates[k].mat
        }
    return rate_params





# ## Setup AMBER

# In[4]:

st_win = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
kinn_model_space = ModelSpace.from_dict([
    # k_01
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='0', TARGET='1', 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
          #RANGE_D=5
     )],
    # k_10
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='1', TARGET='0', 
          EDGE=pmbga.Binomial(alpha=1, beta=1, n=1),
          #EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=10+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),          
          #RANGE_D=5
     )],
    # k_12
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='1', TARGET='2', 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=15+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
          #RANGE_D=5
     )],
    # k_21
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='2', TARGET='1', 
          EDGE=pmbga.Binomial(alpha=1, beta=1, n=1),
          #EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=20+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),             
          #RANGE_D=5
     )],
    # k_23
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='2', TARGET='3', 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=25+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),     
          #RANGE_D=5
     )],
    # k_32
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='3', TARGET='2', 
          EDGE=pmbga.Binomial(alpha=1, beta=1, n=1),        
          #EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=30+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),          
          #RANGE_D=5
     )],
    # k_30
    [dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='3', TARGET='0', 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=35+st_win, prior_cnt=[1]*len(st_win)),
          #RANGE_D=pmbga.Categorical(choices=5+st_win, prior_cnt=[1]*len(st_win)),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
          #RANGE_D=5,
          CONTRIB=1
     )],
    # k_20 [add an optional path to contrib rates w/ prior = 0.5]
    #[dict(Layer_type='conv1d', kernel_size=1, filters=1, SOURCE='2', TARGET='0', 
    #      EDGE= pmbga.Binomial(alpha=1, beta=1, n=1),
    #      RANGE_ST=pmbga.Categorical(choices=20+st_win, prior_cnt=[1]*len(st_win)),
    #      #RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),
    #      RANGE_D=5,
    #      CONTRIB=1
    # )],
])
print(kinn_model_space)


# In[5]:


controller = pmbga.ProbaModelBuildGeneticAlgo(
            model_space=kinn_model_space,
            buffer_type='population',
            buffer_size=50,  # buffer size controlls the max history going back
            batch_size=1,   # batch size does not matter in this case; all arcs will be retrieved
        )


# ## Components before they are implemented in AMBER

# In[6]:


# poorman's manager get reward
def get_reward_pipeline(param_fp):
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)
    with train_graph.as_default(), train_sess.as_default():
        kinn_test = KineticModel(param_fp)
        kinn_test.build_ka_patterns()
        output_op = Operation('dense', activation="sigmoid", units=1, kernel_initializer="zeros", name="output")
        mb = KineticNeuralNetworkBuilder(kinn=kinn_test, session=train_sess, n_feats=50, n_channels=4, output_op=output_op)
        mb.build(optimizer='adam', plot=False)
        #mb.load_data('./data/sim_data/21-11-1_test_1/test_1.csv')
        mb.load_data(args.data_file)
        # train and test
        model = kinn_train_fn(model=mb.model, traindata=mb.traindata, epochs=25, plot=False, verbose=0)
        test_pcc = kinn_test_fn(model=model, testdata=mb.testdata, plot=False)
        # interpret
        rate_params = kinn_interpret_fn(mb)
        rate_df = pd.DataFrame({x : {
            'A': np.round(rate_params[x]['estimate'][0], 3),
            'C': np.round(rate_params[x]['estimate'][1], 3),
            'G': np.round(rate_params[x]['estimate'][2], 3),
            'T': np.round(rate_params[x]['estimate'][3], 3),
        } for x in rate_params }).transpose()
    del train_graph, train_sess
    tf.keras.backend.clear_session()
    return test_pcc , rate_df


# ## A fancy For-Loop that does the work for `amber.architect.trainEnv`

# In[8]:


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

# get prior probas
_, old_probs = compute_eps(controller.model_space_probs)


# In[ ]:


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
            model_params = modelSpace_to_modelParams(arc)
            # get reward
            test_pcc, rate_df = get_reward_pipeline(model_params)
            test_pcc = test_pcc[0]
            # update best, or increase patience counter
            if test_pcc > best_indv:
                best_indv = test_pcc
                has_impr = True
            # store
            _ = controller.store(action=arc, reward=test_pcc)
            hist.append({'gen': generation, 'arc':arc, 'test_pcc': test_pcc, 'rate_df': rate_df})
        end = time.time()
        if generation < n_warmup_gen:
            print(f"Gen {generation} < {n_warmup_gen} warmup.. skipped - Time %.2f" % (end-start), flush=True)
            continue
        if disable_posterior_update is True:
            controller.buffer.finish_path(kinn_model_space, generation, '.')
        else:
            _ = controller.train(episode=generation, working_dir=".")


        delta, old_probs = compute_eps(controller.model_space_probs, old_probs)
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
        pc_cnt = 0 if has_impr else pc_cnt+1
        if pc_cnt >= patience:
            print("early-stop due to max patience w/o improvement")
            break
    except KeyboardInterrupt:
        print("user interrupted")
        break


# In[ ]:


a = pd.DataFrame(hist)
a['arc'] = ['|'.join([f"{x.Layer_attributes['RANGE_ST']}-{x.Layer_attributes['RANGE_ST']+x.Layer_attributes['RANGE_D']}" for x in entry]) for entry in a['arc']]
a.drop(columns=['rate_df'], inplace=True)
a.to_csv(os.path.join(args.wd, "train_history.tsv"), sep="\t", index=False)


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')

ax = stat_df.plot.line(x='Generation', y=['GenAvg', 'Best'])
ax.set_ylabel("Reward (Pearson correlation)")
ax.set_xlabel("Generation")
plt.savefig(os.path.join(args.wd, "reward_over_time.png"))

# In[ ]:



# START SITE
fig, axs_ = plt.subplots(3,3, figsize=(15,15))
axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
for k in controller.model_space_probs:
    if k[-1] == 'RANGE_ST':
        try:
            d = controller.model_space_probs[k].sample(size=1000)
        except:
            continue
        ax = axs[k[0]]
        _ = sns.distplot(d, label="Post", ax=ax)
        _ = sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
        if k[0] < 7:
            _ = ax.axvline(x=kinn_gr.model_params['Rates'][k[0]]['input_range'][0],linestyle='--', color='grey')
            _ = ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d)), 
                          '\nGround truth', str(kinn_gr.model_params['Rates'][k[0]]['input_range'][0])])
            )
        else:
            _ = ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))

        #_ = ax.set_xlim(0,50)
fig.tight_layout()
fig.savefig(os.path.join(args.wd, "range_st.png"))


# In[ ]:


# CONV RANGE
fig, axs_ = plt.subplots(3,3, figsize=(15,15))
axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
for k in controller.model_space_probs:
    if k[-1] == 'RANGE_D':
        d = controller.model_space_probs[k].sample(size=1000)
        ax = axs[k[0]]
        _ = sns.distplot(d, ax=ax)
        _ = sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
        if k[0] < 7:
            D = kinn_gr.model_params['Rates'][k[0]]['input_range'][1] - kinn_gr.model_params['Rates'][k[0]]['input_range'][0]
            _ = ax.axvline(x=D,linestyle='--', color='grey')
            _ = ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d)), '\nGround truth', str(D)])
            )
        else:
            _ = ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
        #_ = ax.set_xlim(0,20)    
fig.tight_layout()
fig.savefig(os.path.join(args.wd, "range_d.png"))

# In[ ]:


# EDGE PRESENCE
fig, axs_ = plt.subplots(3,3, figsize=(15,15))
axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
for k in controller.model_space_probs:
    if k[-1] == 'EDGE':
        d = controller.model_space_probs[k].sample(size=1000)
        ax = axs[k[0]]
        sns.distplot(d, ax=ax)
        sns.distplot(controller.model_space_probs[k].prior_dist, ax=ax)
        ax.set_title(
            ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
        #_ = ax.set_xlim(0,20)    
fig.tight_layout()
fig.savefig(os.path.join(args.wd, "edge.png"))


# In[ ]:


# bogus connections
bogus_conns = [7]
for b in bogus_conns:
    for k in controller.model_space_probs:
        if k[0] != b: continue
        #print(k)
        fig, ax = plt.subplots()
        ax = sns.distplot(controller.model_space_probs[k].sample(size=1000), label="post")
        sns.distplot(controller.model_space_probs[k].prior_dist, ax=ax, label="prior")
        ax.set_title(k)


# In[ ]:




