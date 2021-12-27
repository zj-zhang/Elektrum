#!/usr/bin/env python
# coding: utf-8

# # Probablistic model building genetic algorithm

# In[1]:


#get_ipython().run_line_magic('cd', '/mnt/ceph/users/zzhang/CRISPR_pred/crispr_kinn')


# In[2]:


from src.kinetic_model import KineticModel, modelSpace_to_modelParams
from src.neural_network_builder import KineticNeuralNetworkBuilder

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as ss
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import gc
from sklearn.model_selection import train_test_split


# ## Load data

# In[4]:


x = np.load('./data/compiled_X.npy')
y = np.load('./data/compiled_Y.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777)


# ## Setup AMBER

# In[5]:


import amber
print(amber.__version__)
from amber.architect import pmbga
from amber.architect import ModelSpace, Operation


# In[6]:


kinn_model_space = ModelSpace.from_dict([
    # k_01, sol -> open R-loop
    [dict(Layer_type='conv1d', filters=1, SOURCE='0', TARGET='1', 
          kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=[1]*3),
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=[0,1,2,3,4], prior_cnt=[1]*5),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
     )],
    # k_10, open R-loop -> sol
    [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='0', 
          kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=[1]*3),
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=[0,1,2,3,4], prior_cnt=[1]*5),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),          
     )],
    # k_12, open R-loop -> intermediate R-loop
    [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='2', 
          kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=[1]*3), 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=[5,6,7,8,9,10], prior_cnt=[1]*6),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
     )],
    # k_21, intermediate R-loop -> open R-loop
    [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='1', 
          kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=[1]*3), 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=[5,6,7,8,9,10], prior_cnt=[1]*6),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),             
     )],
    # k_23, intermediate R-loop -> closed R-loop
    [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='3', 
          kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=[1]*3), 
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=[11,12,13,14,15,16,17,18], prior_cnt=[1]*8),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),     
     )],
    # k_32
    [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='2', 
          kernel_size=pmbga.Categorical(choices=[1,2,3], prior_cnt=[1]*3),
          EDGE=1,        
          RANGE_ST=pmbga.Categorical(choices=[11,12,13,14,15,16,17,18], prior_cnt=[1]*8),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),          
     )],
    # k_30
    [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='0', 
          kernel_size=pmbga.Categorical(choices=[1,2,3,4,5,6], prior_cnt=[1]*6),
          EDGE=1,
          RANGE_ST=pmbga.Categorical(choices=np.arange(0,19), prior_cnt=[1]*19),
          RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
          CONTRIB=1
     )],
])
print(kinn_model_space)


# In[7]:


controller = pmbga.ProbaModelBuildGeneticAlgo(
            model_space=kinn_model_space,
            buffer_type='population',
            buffer_size=50,  # buffer size controlls the max history going back
            batch_size=1,   # batch size does not matter in this case; all arcs will be retrieved
        )


# ## Components before they are implemented in AMBER

# In[8]:


## NEEDS RE-WORK
# poorman's manager get reward
def get_reward_pipeline(model_arcs):
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)
    model_params = modelSpace_to_modelParams(model_arcs)
    tf.reset_default_graph()
    with train_graph.as_default(), train_sess.as_default():
        kinn_test = KineticModel(model_params)
        mb = KineticNeuralNetworkBuilder(kinn=kinn_test, session=train_sess, n_channels=13)
        # train and test
        mb.build(optimizer='adam', plot=False, output_act=False)
        model = mb.model
        x_train_b = mb.blockify_seq_ohe(x_train)
        x_test_b = mb.blockify_seq_ohe(x_test)
        checkpointer = ModelCheckpoint(filepath="bestmodel.h5", mode='min', verbose=0, save_best_only=True,
                               save_weights_only=True)
        earlystopper = EarlyStopping(
            monitor="val_loss",
            mode='min',
            patience=5,
            verbose=0)

        hist = model.fit(x_train_b, y_train[:,1],
                  batch_size=768,
                  validation_split=0.2,
                  callbacks=[checkpointer, earlystopper],
                  epochs=5, verbose=0)
        y_hat = model.predict(x_test_b).flatten()
        test_pcc = ss.pearsonr(y_hat, y_test[:,1])[0]
    del train_graph, train_sess
    del model, hist
    tf.keras.backend.clear_session() # THIS IS IMPORTANT!!!
    gc.collect()
    return test_pcc


# ## A fancy For-Loop that does the work for `amber.architect.trainEnv`

# In[9]:


# trainEnv parameters
samps_per_gen = 10   # how many arcs to sample in each generation; important
max_gen = 1500
epsilon = 0.05
patience = 500
n_warmup_gen = -1


# In[10]:


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


# In[11]:


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
                test_pcc = get_reward_pipeline(arc)
            except ValueError:
                test_pcc = 0
            except Exception as e:
                raise e
            rate_df = None
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


# In[12]:


pd.DataFrame(hist).sort_values('test_pcc', ascending=False)


# In[13]:


a = pd.DataFrame(hist)
a['arc'] = ['|'.join([f"{x.Layer_attributes['RANGE_ST']}-{x.Layer_attributes['RANGE_ST']+x.Layer_attributes['RANGE_D']}" for x in entry]) for entry in a['arc']]
a.drop(columns=['rate_df'], inplace=True)
a.to_csv("train_history.tsv", sep="\t", index=False)


# In[21]:


#get_ipython().run_line_magic('matplotlib', 'inline')

ax = stat_df.plot.line(x='Generation', y=['GenAvg', 'Best'])
ax.set_ylabel("Reward (Pearson correlation)")
ax.set_xlabel("Generation")
plt.savefig("reward_vs_time.png")


# In[20]:


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
        sns.distplot(d, label="Post", ax=ax)
        sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
        ax.set_title(
            ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))

        #_ = ax.set_xlim(0,50)
fig.tight_layout()
fig.savefig("range_st.png")


# In[22]:


# CONV RANGE
fig, axs_ = plt.subplots(3,3, figsize=(15,15))
axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
for k in controller.model_space_probs:
    if k[-1] == 'RANGE_D':
        d = controller.model_space_probs[k].sample(size=1000)
        ax = axs[k[0]]
        sns.distplot(d, ax=ax)
        sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
        ax.set_title(
                ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
        #_ = ax.set_xlim(0,20)    
fig.tight_layout()
fig.savefig("range_d.png")


# In[19]:


# KERNEL SIZE 
fig, axs_ = plt.subplots(3,3, figsize=(15,15))
axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
for k in controller.model_space_probs:
    if k[-1] == 'kernel_size':
        d = controller.model_space_probs[k].sample(size=1000)
        ax = axs[k[0]]
        sns.distplot(d, ax=ax)
        sns.distplot(controller.model_space_probs[k].prior_dist, ax=ax)
        ax.set_title(
            ' '.join(['Rate ID', str(k[0]), '\nPosterior mean', str(np.mean(d))]))
        #_ = ax.set_xlim(0,20)    
fig.tight_layout()
fig.savefig("kernel_size.png")

# In[ ]:




