from src.kinetic_model import KineticModel
from src.neural_network_builder import KineticNeuralNetworkBuilder, KineticEigenModelBuilder
import sys
import time
import pickle
import os
from amber.utils import corrected_tf as tf
import numpy as np
from amber.utils import run_from_ipython
import copy

def reload_from_dir(wd, manager_kwargs, sess=None, model_fn=None, load_weights=True, verbose=False):
    # unpack keyword args
    n_channels = manager_kwargs.get("n_channels", 9)
    n_feats = manager_kwargs.get("n_feats", 25)
    replace_conv_by_fc = manager_kwargs.get("replace_conv_by_fc", False)
    opt = manager_kwargs.get("optimizer", None)
    output_op = manager_kwargs.get("output_op", None)

    model_params = pickle.load(open(os.path.join(wd, "AmberSearchBestModel_config.pkl"), "rb"))
    kinn = KineticModel(model_params)
    model_fn =  model_fn or KineticNeuralNetworkBuilder
    mb = model_fn(kinn=kinn, session=sess,
            output_op=output_op,
            n_feats=n_feats,
            n_channels=n_channels, 
            replace_conv_by_fc=replace_conv_by_fc
    )
    opt = opt() if opt else 'adam'
    mb.build(optimizer=opt, plot=False, output_act=False)
    if load_weights is True:
        if verbose: print("loaded searched model")
        mb.model.load_weights(os.path.join(wd, "AmberSearchBestModel.h5"))
    #mb.model.summary()
    return mb


def get_rate_model_from_kinn(kinn):
    layer_dict = {l.name:l for l in kinn.model.layers}
    rate_mod = tf.keras.models.Model(inputs=kinn.model.inputs, outputs=layer_dict['gather_rates'].output)
    matched = np.zeros((1,25,9))
    rates = rate_mod.predict(kinn.blockify_seq_ohe(matched))
    return rate_mod, rates


def retrain_last_layer(wd, manager_kwargs, model_fn, new_output_op, new_name_suffix, datas=None, sess=None):
    """
    Example
    -------
    .. code: python
        from silence_tensorflow import silence_tensorflow; silence_tensorflow()
        import os
        import tensorflow as tf
        from src.neural_network_builder import KineticNeuralNetworkBuilder
        from src.reload import retrain_last_layer
        from src.data import load_finkelstein_data as get_data
        wds = ["./outputs/final/%s"%x for x in os.listdir("outputs/final/") if x.endswith("gRNA1")]
        datas = get_data(target="wtCas9_cleave_rate_log", make_switch=False, logbase=10)
        manager_kwargs = {'n_feats': 25, 'n_channels': 9,}
        new_output_op = lambda: tf.keras.layers.Dense(units=1, activation="linear", name="output_nonneg", kernel_constraint=tf.keras.constraints.NonNeg())
        for wd in wds:
            print(wd)
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                retrain_last_layer(wd=wd, manager_kwargs=manager_kwargs, new_output_op=new_output_op, new_name_suffix="linear_offset", model_fn=KineticNeuralNetworkBuilder, datas=datas)
    """
    manager_kw2 = copy.copy(manager_kwargs)
    manager_kw2['output_op'] = new_output_op
    kinn = reload_from_dir(wd=wd, manager_kwargs=manager_kw2, model_fn=model_fn,
            load_weights=False, sess=sess)
    if datas is not None:
        (x_train, y_train), (x_test, y_test) = datas
        t0 = time.time()
        checkpointer = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join(wd, f"bestmodel_{new_name_suffix}.h5"), mode='min', verbose=0, save_best_only=True, save_weights_only=True)
        earlystopper = tf.keras.callbacks.EarlyStopping( monitor="val_loss", mode='min', patience=50, verbose=0)
        x_train_b = kinn.blockify_seq_ohe(x_train)
        x_test_b = kinn.blockify_seq_ohe(x_test)

        hist = kinn.model.fit(
            x_train_b, y_train,
            epochs=3000,
            batch_size=128,
            validation_data=[x_test_b, y_test],
            callbacks=[checkpointer, earlystopper],
            verbose=0
        )
        print("training took %.3f secs.." % (time.time()-t0))
        kinn.model.load_weights(os.path.join(wd, f"bestmodel_{new_name_suffix}.h5"))
    return kinn




if __name__ == "__main__" and not run_from_ipython():
    wd = sys.argv[1]
    reload_from_dir(wd=wd)
