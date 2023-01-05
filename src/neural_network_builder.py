"""class for converting a kinetic model hypothesis to a keras neural network
"""

from amber.utils import corrected_tf as tf
import amber
import tensorflow as tf2
from tensorflow.keras.layers import Input, Conv1D, Dense, Concatenate, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from src.kinetic_model import KineticModel

from amber.modeler.kerasModeler import ModelBuilder
from amber.modeler.dag import get_layer


# TODO 
class KineticRate(tf.keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, kernel_initializer='zeros',):
        super().__init__()


class KineticNeuralNetworkBuilder(ModelBuilder):
    def __init__(self, kinn:KineticModel, session:tf.Session=None, 
                 output_op:Union[amber.architect.Operation, None]=None, 
                 n_feats=25, n_channels=4, replace_conv_by_fc=False):
        """Convert a kinetic state graph (with state-specific input sequence ranges) to a neural network,
        whose complexity is specified in rate_pwm_len

        Parameters
        ----------
        kinn : KineticModel
            reference to the biophysics model class
        session : tf.Session
            underlying session for tensorflow and keras
        output_op : amber.architect.Operation, or None
            specific operation for converting activity to phenotype; 
            if None, will learn from data by a Dense layer
        n_feats: int
            TODO What are these related to
        n_channels : int
            TODO What are these related to
        replace_conv_by_fc : bool
            TODO What does this do?

        Attributes
        ----------
        kinn : KineticModel
            Stored kinn
        rate_pwm_len : list
            list of pwm lengths. Together with kinn, the two variable that controls the neural network
        session : tf.Session
            Stored session
        output_op : amber.architect.Operation, or None
            Stored output_op
        model : tf.keras.Model, or None
            keras model passing through the internal objects; will be None before initialized
        layer_dict : dict
            dictionary that maps layer name string to layer objects
        input_ranges : list of tuples
            a tuple of a pair of integers, ordered by the model.inputs
        traindata : np.ndarray, or None
        testdata : np.ndarray, or None

        """
        self.kinn = kinn

        # check rate_mode
        rate_pwm_len = [d.kernel_size for d in self.kinn.rates]
        assert len(rate_pwm_len) == len(self.kinn.rates)
        for pwm_len, rate in zip(*[rate_pwm_len, self.kinn.rates]):
            assert isinstance(pwm_len, int)
        self.rate_pwm_len = rate_pwm_len

        # start new session
        self.session = session
        if self.session is None:
            self._reset_session()
        else:
            tf.keras.backend.set_session(self.session)
        self.weight_initializer = 'zeros'

        # placeholders
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.replace_conv_by_fc = replace_conv_by_fc
        self.output_op = output_op
        self.model = None
        self.layer_dict = None
        self.input_ranges = None
        self.traindata = None
        self.testdata = None

    def _reset_session(self):
        if self.session is not None:
            self.session.close()
        self.session = tf.Session()
        tf.keras.backend.set_session(self.session)

    def _build_inputs(self):
        """TODO what does this do?

        Returns
        -------
        _type_
            _description_
        """
        inputs_op = {}
        self.input_ranges = []
        for a, b in set([tuple(r.input_range) for r in self.kinn.rates]):
            assert a >= 0
            b = min(b, self.n_feats-1)
            input_id = "input_%i_%i" % (a, b)
            if input_id not in inputs_op:
                assert b-a > 0, ValueError(f"input range less than 0 for {input_id}")
                inputs_op["input_%i_%i" % (a, b)] = Input(
                    shape=(b - a, self.n_channels), name="input_%i_%i" % (a, b))
                self.input_ranges.append((a, b))
        return inputs_op

    def _build_rates(self, inputs_op):
        # build convs --> rates
        rates = []
        for i, rate in enumerate(self.kinn.rates):

            seq_range = f"input_{rate.input_range[0]}_{min(self.n_feats-1, rate.input_range[1])}"
            name = "k%i" % i

            seq_range_d = min(self.n_feats-1,rate.input_range[1]) - rate.input_range[0]

            if self.replace_conv_by_fc: # if replace is True, overwrite the rate dict
                padding = "valid"
            else:
                padding = rate.__dict__.get("padding", "valid" if self.replace_conv_by_fc else "same")
            filters = rate.__dict__.get("filters", 1)

            conv_rate = Conv1D(filters=filters,
                           kernel_size=(seq_range_d,) if self.replace_conv_by_fc else self.rate_pwm_len[i],
                           activation="linear",
                           #use_bias=False,
                           kernel_initializer=self.weight_initializer() if callable(self.weight_initializer) else self.weight_initializer,
                           padding=padding,
                           name="conv_%s" % name)(
                        inputs_op[seq_range])
            hidden_size = rate.__dict__.get("hidden_size", 0)

            if hidden_size == 0:
                conv_rate = Flatten()(conv_rate)
                rate = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name="sum_%s" % name)(conv_rate)
            else:
                reshape_fn = rate.__dict__.get("reshape_fn", 0)
                reshape_map = {0: tf.keras.layers.Flatten, 1:tf.keras.layers.GlobalAveragePooling1D, 2:tf.keras.layers.GlobalMaxPooling1D}
                conv_rate = reshape_map[reshape_fn]()(conv_rate)
                h = Dense(units=hidden_size, activation='relu', 
                        kernel_initializer=self.weight_initializer() if callable(self.weight_initializer) else self.weight_initializer,
                        name="hidden_%s"%name,
                        )(conv_rate)
                rate = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name="sum_%s" % name)(h)

            rates.append(rate)
        return rates

    def _build_king_altman(self, rates):
        concat = Concatenate(name="gather_rates")(rates)
        king_altman = Lambda(lambda x: tf.nn.softmax(
            tf.matmul(x, tf.constant(self.kinn.get_ka_pattern_mat().transpose(), dtype=tf.float32))),
            name="KingAltman")(concat)
        return concat, king_altman

    def _build_activity(self, rates, king_altman):
        # build activity
        activity = []
        rate_contrib_mat = self.kinn.get_rate_contrib_matrix()
        for k in range(rate_contrib_mat.shape[1]):
            # get each column as mask
            mask = rate_contrib_mat[:, k]
            assert np.sum(mask) <= 1, "k=%i, mask error for %s" % (k, mask)
            if np.sum(mask) == 0:
                continue
            rate_index = np.where(mask == 1)[0][0]
            rate_layer = rates[rate_index]
            rate_layer = Lambda(
                lambda x: tf.math.exp(x),
                name=f'exp_k{rate_index}_{k}')(rate_layer)
            ka_slice = Lambda(
                lambda x: tf.gather(
                    x, [k], axis=-1), name=f"KA_slice_{k}")(king_altman)
            intermediate = Concatenate(name=f"gather_act_{k}")([
                ka_slice, rate_layer])
            activity.append(
                Lambda(
                    lambda x: tf.reduce_prod(x, axis=-1, keepdims=True),
                    name=f"prod_act_{k}")(intermediate)
            )
        return activity

    def _build_outputs(self, activity):
        # build outputs
        # TODO: change the hard-coded output
        if len(activity) > 1:
            if self.output_op is None:
                output = Dense(units=1, activation="linear",
                           kernel_initializer='zeros', 
                           name="output")(
                    Concatenate()(activity))
            else:
                x = Concatenate()(activity)
                output = get_layer(x=x, state=self.output_op, with_bn=False)
        else:
            if self.output_op is None:
                output = Dense(
                    units=1,
                    activation="linear",
                    kernel_initializer='zeros',
                    name="output")(
                    activity[0])
            else:
                output = get_layer(x=activity[0], state=self.output_op, with_bn=False)

        return output

    def build(self, optimizer=None, output_act=False, plot=False, return_intermediate=False):
        """build the machine learning model

        Parameters
        ----------
        optimizer : str, dict, or keras.optimizer
            optimizer to use, can take various forms. If left None, then will use SGD with a 
            learning rate of 0.1 and momentum 0.95
        output_act : bool
            if true, output activity instead of phenotype
        """
        inputs_op = self._build_inputs()
        rates = self._build_rates(inputs_op)
        gather_rates, king_altman = self._build_king_altman(rates)
        activity = self._build_activity(rates, king_altman)
        if return_intermediate:
            return {
                'inputs_op': inputs_op, 
                'gather_rates': gather_rates, 
                'KingAltman': king_altman, 
                'activity': activity}
        optimizer = optimizer or SGD(lr=0.1, momentum=0.95, decay=1e-5, clipnorm=1.0)
        if output_act is True:
            self.model = Model([inputs_op[j] for j in inputs_op], [
                               activity[k] for k in range(len(activity))])
            self.model.compile(
                loss='mse',
                optimizer=optimizer
            )
        else:
            output = self._build_outputs(activity)
            self.model = Model([inputs_op[j] for j in inputs_op], output)
            self.model.compile(
                # TODO
                #loss='binary_crossentropy',
                loss='mse',
                optimizer=optimizer
            )
        self.layer_dict = {l.name: l for l in self.model.layers}
        if plot is True:
            plot_model(self.model, to_file='model.png')

    def load_data(self, fp, output_act=False):
        """populate the instance's data attributes for training and testing

        Parameters
        ----------
        fp : str
            filepath for data csv
        output_act : bool
            if true, use activity (currently hard-coded as column 50) as output; otherwise use phenotype
            (currently hard-coded as column 51) as output
            Activity will be normalized by max value
        """
        # one-hot encoder
        x_set = [['A'], ['C'], ['G'], ['T']]
        enc = OneHotEncoder(sparse=False, categories='auto')
        _ = enc.fit(x_set)

        # read in data
        with open(fp, 'r') as f:
            data = pd.read_csv(f, index_col=0)
        data.dropna(inplace=True)
        data['seq_ohe'] = [enc.transform(
            [[t] for t in row[1][0:50]]) for row in data.iterrows()]
        if output_act is True:
            data['obs'] = data['50']
            data['obs'] /= data['obs'].max()
        else:
            data['obs'] = data['51']

        # split train-test
        gen_df = data
        X_train, X_test, y_train, y_test = train_test_split(
            gen_df['seq_ohe'].values, gen_df['obs'].values, test_size=0.2, random_state=777)
        input_seqs_ohe = []
        for i in range(len(X_train)):
            input_seqs_ohe += [X_train[i]]
        test_seqs_ohe = []
        for i in range(len(X_test)):
            test_seqs_ohe += [X_test[i]]

        # blockify
        input_seqs_ohe = np.array(input_seqs_ohe)
        test_seqs_ohe = np.array(test_seqs_ohe)
        x_train_b = self.blockify_seq_ohe(input_seqs_ohe)
        x_test_b = self.blockify_seq_ohe(test_seqs_ohe)
        self.traindata = x_train_b, y_train, input_seqs_ohe
        self.testdata = x_test_b, y_test, test_seqs_ohe


    def blockify_seq_ohe(self, seq_ohe):
        """separate seq matrices into blocks according to input_ranges

        Parameters
        ----------
        seq_ohe : np.array
            a np.ndarray that stores one-hoe encoded sequences

        Returns
        -------
        list
            each element is a np.array used as input that matches input_ranges
        """
        data = []
        for a, b in self.input_ranges:
            data.append(seq_ohe[:, a:b])
        return data

    def predict(self, data):
        assert isinstance(data, np.ndarray)
        blocks = self.blockify_seq_ohe(data)
        return self.model.predict(blocks)


class KineticMatrixLayer(tf.keras.layers.Layer):
    """Custom Layer that converts kinetic rates into a kinetic matrix
    """
    def __init__(self, num_rates: int, num_states: int, scatter_ind: list, 
                 regularize_eigenval_ratio : float, top_k_eigvals=3):
        super().__init__()
        self.num_rates = num_rates
        self.num_states = num_states
        self.scatter_ind = scatter_ind
        self.top_k_eigvals = top_k_eigvals
        self.regularize_eigenval_ratio = regularize_eigenval_ratio

    def build(self, *args):
        super().build(*args)

    @tf.autograph.experimental.do_not_convert
    def _scatter_single_sample(self, input_tensor):
        updates = []
        inds = []
        assert input_tensor.shape[0] == self.num_rates
        for i in range(self.num_rates):
            val = input_tensor[i]
            for ind, sign in self.scatter_ind[i]:
                updates.append(val*sign)
                inds.append(ind)
        kinetic_matrix = tf2.scatter_nd(inds, updates, (self.num_states, self.num_states))
        eigvals = tf2.math.real(tf2.linalg.eigvals(kinetic_matrix))
        return eigvals

    def call(self, inputs: tf.Tensor):
        batch_eigvals = tf2.map_fn(self._scatter_single_sample, inputs)
        batch_top_ev = tf2.math.top_k(batch_eigvals, k=self.top_k_eigvals, sorted=True)
        if self.regularize_eigenval_ratio > 0 :
            # all eigvals <= 0; larger is ~0,  smaller is more negative
            # minimize loss -> lam_1 / lam_2 ~ 0 
            self.add_loss(self.regularize_eigenval_ratio *
                    tf.reduce_mean(tf.math.divide_no_nan(batch_top_ev.values[:,1], batch_top_ev.values[:,2])))
        return batch_top_ev.values


class KineticEigenModelBuilder(KineticNeuralNetworkBuilder):
    def __init__(self, kinn, session=None, output_op=None, 
                 n_feats=25, n_channels=4, replace_conv_by_fc=False,
            regularize_eigenval_ratio=0.001):
        super().__init__(kinn=kinn, session=session, output_op=output_op, n_feats=n_feats, n_channels=n_channels, 
                replace_conv_by_fc=replace_conv_by_fc)
        #self.weight_initializer = 'random_normal'
        self.weight_initializer = lambda: tf.keras.initializers.RandomNormal(stddev=0.01)
        self.regularize_eigenval_ratio = regularize_eigenval_ratio

    # overwrite activity without going through KA
    def _build_activity(self,rates):
        concat = tf2.keras.layers.Concatenate(name='gather_rates')(rates)
        concat = tf2.keras.layers.Lambda(lambda x: tf2.exp(x), name="exp_rates")(concat)
        scatter_ind = [r.scatter_nd for r in self.kinn.rates]
        kinetic_eig = KineticMatrixLayer(num_rates=len(rates), num_states=len(self.kinn.states),
                scatter_ind=scatter_ind, regularize_eigenval_ratio=self.regularize_eigenval_ratio)(concat)
        return [kinetic_eig]


    # overwrite build to ignore KA
    def build(self, optimizer=None, output_act=False, plot=False):
        inputs_op = self._build_inputs()
        rates = self._build_rates(inputs_op)
        activity = self._build_activity(rates,)
        optimizer = optimizer or tf2.keras.optimizers.SGD(lr=0.1, momentum=0.95, decay=1e-5)
        if output_act is True:
            self.model = tf2.keras.models.Model([inputs_op[j] for j in inputs_op], [
                               activity[k] for k in range(len(activity))])
            self.model.compile(
                loss='mae',
                optimizer=optimizer
            )
        else:
            output = self._build_outputs(activity)
            self.model = tf2.keras.models.Model([inputs_op[j] for j in inputs_op], output)
            self.model.compile(
                # TODO
                #loss='binary_crossentropy',
                loss='mae',
                optimizer=optimizer
            )
        self.layer_dict = {l.name: l for l in self.model.layers}
        if plot is True:
            plot_model(self.model, to_file='model.png')




