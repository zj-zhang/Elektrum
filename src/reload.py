from src.kinetic_model import KineticModel
from src.neural_network_builder import KineticNeuralNetworkBuilder, KineticEigenModelBuilder
import sys
import pickle
import os
from amber.utils import corrected_tf as tf
import numpy as np
from amber.utils import run_from_ipython

def reload_from_dir(wd, manager_kwargs, sess=None, model_fn=None):
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
    mb.model.load_weights(os.path.join(wd, "AmberSearchBestModel.h5"))
    #mb.model.summary()
    return mb


if __name__ == "__main__" and not run_from_ipython():
    wd = sys.argv[1]
    reload_from_dir(wd=wd)
