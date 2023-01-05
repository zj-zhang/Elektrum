
from distutils.sysconfig import customize_compiler
import tensorflow as tf
import numpy as np
import os
from src.data import load_finkelstein_data as get_data
from src.transfer_learn import KinnLayer
from src.reload import reload_from_dir


def test_transfer():
    # test
    include_ref = True
    seq = tf.keras.layers.Input((25, 13), name='seq')
    x = tf.keras.layers.GlobalAveragePooling1D()(seq)
    x = tf.keras.layers.Dense(units=10)(x)
    out = KinnLayer(
        kinn_dir="test_files/KINN-finkelstein-gRNA1_test/",
        manager_kws={'output_op': lambda: tf.keras.layers.Lambda(
            lambda x: tf.math.log(x)/np.log(10), name="output")},
        channels=np.arange(4, 13),
        kinn_trainable=False
    )([x, seq])
    model = tf.keras.models.Model(inputs=seq, outputs=out)
    try:
        session = tf.keras.backend.get_session()
    except:
        session = tf.compat.v1.keras.backend.get_session()
    kinn_indep = reload_from_dir(
        workdir="test_files/KINN-finkelstein-gRNA1_test/",
        manager_kwargs={'output_op': lambda: tf.keras.layers.Lambda(
            lambda x: tf.math.log(x)/np.log(10), name="output")},
        sess=session
    )

    (x_train, y_train), (x_test, y_test) = get_data(logbase=10, include_ref=True)

    # test forward
    print("FORWARD")
    print(model.predict(x_test[0:10]))
    print('-'*10)

    kinn_before_train = model.layers[-1].kinn_header.predict(
        model.layers[-1].mb.blockify_seq_ohe(x_test[0:3, :, 4:]))
    kinn_indep_pred = kinn_indep.model.predict(
        model.layers[-1].mb.blockify_seq_ohe(x_test[0:10, :, 4:]))
    print(kinn_indep_pred)
    print('-'*10)

    # test backward
    print("BACKWARD")
    model.compile(loss='mse', optimizer='adam')
    losses = []
    for _ in range(15):
        losses.append(model.train_on_batch(x=x_test[0:3], y=np.ones((3, 1))))

    print(losses)
    print(model.predict(x_test[0:3]))

    kinn_after_train = model.layers[-1].kinn_header.predict(
        model.layers[-1].mb.blockify_seq_ohe(x_test[0:3, :, 4:]))
    assert np.array_equal(kinn_before_train, kinn_after_train), "KINN changed"

    # test save & load
    print("SAVE & LOAD")
    model.save_weights('test.h5')
    model.load_weights('test.h5')
    model.save('test.h5')
    # not passing
    #model = tf.keras.models.load_model('test.h5', custom_objects={'KinnLayer': KinnLayer})
