from src.kinetic_model import KineticModel, modelSpace_to_modelParams
from src.neural_network_builder import KineticNeuralNetworkBuilder, KineticEigenModelBuilder
import tensorflow.compat.v1 as tf1
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os, sys, shutil, pickle, gc, time
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ## A For-Loop that does the work for `amber.architect.trainEnv`
def search_env(controller, wd, evo_params, manager_kwargs=None, disable_posterior_update=False):
    manager_kwargs = manager_kwargs or {}
    # unpack evo params
    samps_per_gen = evo_params.get("samps_per_gen", 10)
    max_gen = evo_params.get("max_gen", 200)
    patience = evo_params.get("patience", 20)
    n_warmup_gen = evo_params.get("n_warmup_gen", 0)
    model_fn = evo_params.get("model_fn", KineticNeuralNetworkBuilder)
    output_op = evo_params.get("output_op", None)
    x_train, y_train = evo_params.get("train_data", (None, None))
    x_test, y_test = evo_params.get("test_data", (None, None))
    # variables to be filled
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
                            wd=wd,
                            model_fn=model_fn,
                            **manager_kwargs
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
                    shutil.move(os.path.join(wd, "bestmodel.h5"), os.path.join(wd, "AmberSearchBestModel.h5"))
                    shutil.move(os.path.join(wd, "model_params.pkl"), os.path.join(wd, "AmberSearchBestModel_config.pkl"))
                # store
                _ = controller.store(action=arc, reward=test_reward)
                hist.append({'gen': generation, 'arc':arc, 'test_reward': test_reward, 'rate_df': rate_df})
            end = time.time()
            if generation < n_warmup_gen:
                print(f"Gen {generation} < {n_warmup_gen} warmup.. skipped - Time %.2f" % (end-start), flush=True)
                continue
            if disable_posterior_update is True:
                controller.buffer.finish_path(controller.model_space, generation, working_dir=wd)
            else:
                _ = controller.train(episode=generation, working_dir=wd)
            post_vars = [np.var(x.sample(size=100)) for _, x in controller.model_space_probs.items()]
            stat_df = stat_df.append({
                'Generation': generation,
                'GenAvg': controller.buffer.r_bias,
                'Best': best_indv,
                'PostVar': np.mean(post_vars)
            }, ignore_index=True)
            print("[%s] Gen %i - Mean fitness %.3f - Best %.4f - PostVar %.3f - Time %.2f" % (
                datetime.now().strftime("%H:%M:%S"),
                generation, 
                controller.buffer.r_bias, 
                best_indv, 
                np.mean(post_vars),
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
    a.to_csv(os.path.join(wd,"train_history.tsv"), sep="\t", index=False)
    ax = stat_df.plot.line(x='Generation', y=['GenAvg', 'Best'])
    ax.set_ylabel("Reward (Pearson correlation)")
    ax.set_xlabel("Generation")
    plt.savefig(os.path.join(wd, "reward_vs_time.png"))
    
    # save
    pickle.dump(controller, open(os.path.join(wd, "controller_states.pkl"), "wb"))

    # plot
    make_plots(controller, canvas_nrow=np.ceil(np.sqrt(len(controller.model_space))), wd=wd)
    return controller, hist, stat_df


# poorman's manager get reward
def get_reward_pipeline(model_arcs, x_train, y_train, x_test, y_test, wd, model_fn=None, **kwargs):
    # unpack keyword args
    n_channels = kwargs.get("n_channels", 9)
    n_feats = kwargs.get("n_feats", 25)
    replace_conv_by_fc = kwargs.get("replace_conv_by_fc", False)
    opt = kwargs.get("optimizer", None)
    output_op = kwargs.get("output_op", None)
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    tf1.logging.set_verbosity(tf1.logging.ERROR)
    train_graph = tf1.Graph()
    train_sess = tf1.Session(graph=train_graph)
    model_params = modelSpace_to_modelParams(model_arcs)
    pickle.dump(model_params, open(os.path.join(wd, "model_params.pkl"), "wb"))
    tf1.reset_default_graph()
    with train_graph.as_default(), train_sess.as_default():
        kinn_test = KineticModel(model_params)
        mb = model_fn(
                kinn=kinn_test,
                session=train_sess,
                output_op=output_op,
                n_channels=n_channels,
                n_feats=n_feats,
                replace_conv_by_fc=replace_conv_by_fc
        )
        # train and test
        opt = opt() if opt else "adam"
        mb.build(optimizer=opt, plot=False, output_act=False)
        model = mb.model
        x_train_b = mb.blockify_seq_ohe(x_train)
        x_test_b = mb.blockify_seq_ohe(x_test)
        checkpointer = ModelCheckpoint(filepath=os.path.join(wd,"bestmodel.h5"), mode='min', verbose=0, save_best_only=True,
                               save_weights_only=True)
        earlystopper = EarlyStopping(
            monitor="val_loss",
            mode='min',
            patience=kwargs.get("earlystop", 5),
            verbose=0)
        try:
            hist = model.fit(x_train_b, y_train,
                      batch_size=kwargs.get("batch_size", 128),
                      validation_data=(x_test_b, y_test),
                      callbacks=[checkpointer, earlystopper],
                      epochs=kwargs.get("epochs", 20),
                      verbose=kwargs.get("verbose", 0))
            model.load_weights(os.path.join(wd,"bestmodel.h5"))
            y_hat = model.predict(x_test_b).flatten()
            reward_fn = kwargs.get("reward_fn", lambda y_hat, y_test: ss.pearsonr(y_hat, y_test)[0])
            test_reward = reward_fn(y_hat, y_test)
        except tf.errors.InvalidArgumentError as e: # eigen could fail
            test_reward = np.nan
            #raise e
        #except ValueError:
        #    test_reward = np.nan
        #test_reward = ss.spearmanr(y_hat, y_test).correlation
        if np.isnan(test_reward):
            test_reward = 0
    del train_graph, train_sess
    del model
    tf.keras.backend.clear_session() # THIS IS IMPORTANT!!!
    gc.collect()
    return test_reward


def make_plots(controller, canvas_nrow, wd):
    canvas_nrow = int(canvas_nrow)
    tot_distr = set([k[-1] for k in controller.model_space_probs])
    for distr_key in tot_distr:
        fig, axs_ = plt.subplots(canvas_nrow, canvas_nrow, figsize=(4.5*canvas_nrow,4.5*canvas_nrow))
        axs = [axs_[i][j] for i in range(len(axs_)) for j in range(len(axs_[i]))]
        for k in controller.model_space_probs:
            if k[-1] == distr_key:
                try:
                    d = controller.model_space_probs[k].sample(size=1000)
                except:
                    continue
                ax = axs[k[0]]
                sns.distplot(d, label="Post", ax=ax)
                sns.distplot(controller.model_space_probs[k].prior_dist, label="Prior", ax=ax)
                ax.set_title(
                    ' '.join(['Rate ID', str(k[0]), '\nPosterior mode', str(ss.mode(d).mode[0])]))
        fig.suptitle(distr_key)
        fig.tight_layout()
        fig.savefig(os.path.join(wd, f"{distr_key}.png"))



