from src.kinetic_model import KineticModel
from src.neural_network_builder import KineticNeuralNetworkBuilder
import sys
import pickle
import os
from amber.utils import run_from_ipython

def reload_from_dir(wd, replace_conv_by_fc=True, n_channels=13):
    model_params = pickle.load(open(os.path.join(wd, "AmberSearchBestModel_config.pkl"), "rb"))
    kinn = KineticModel(model_params)
    mb = KineticNeuralNetworkBuilder(kinn=kinn, n_channels=n_channels, replace_conv_by_fc=replace_conv_by_fc)
    mb.build(optimizer="adam", plot=False, output_act=False)
    mb.model.load_weights(os.path.join(wd, "AmberSearchBestModel.h5"))
    mb.model.summary()
    return mb


if __name__ == "__main__" and not run_from_ipython():
    wd = sys.argv[1]
    reload_from_dir(wd=wd)
