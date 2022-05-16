from amber.architect import pmbga
from amber.architect import ModelSpace, Operation
import numpy as np

# Model Space
def get_informed_ms():
    """model space based on https://www.biorxiv.org/content/10.1101/2020.05.21.108613v2
    """
    #ks_choices=[1,3,7]
    #default_ks = lambda: pmbga.Categorical(choices=ks_choices, prior_cnt=1)
    default_ks = lambda: 1
    st_noise = np.arange(10) - 5
    default_noise_st = lambda a: pmbga.Categorical(choices=np.clip(a+st_noise, 0, 45), prior_cnt=1)
    default_d = lambda: pmbga.Categorical(choices=np.arange(2,8), prior_cnt=1)
    kinn_model_space = ModelSpace.from_dict([
        # k_on, sol -> open R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='0', TARGET='1',
              kernel_size=default_ks(),
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(5),
              RANGE_D=default_d(),
         )],
        # k_off, open R-loop -> sol
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='0', 
              kernel_size=default_ks(),
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(8),
              RANGE_D=default_d()
         )],
        # k_OI, open R-loop -> intermediate R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='2', 
              kernel_size=default_ks(), 
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(10),
              RANGE_D=default_d()
         )],
        # k_IO, intermediate R-loop -> open R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='1', 
              kernel_size=default_ks(), 
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(14),
              RANGE_D=default_d()
         )],
        # k_IC, intermediate R-loop -> closed R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='3', 
              kernel_size=default_ks(), 
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(18),
              RANGE_D=default_d()
         )],
        # k_CI, closed R-loop -> intermediate R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='2', 
              kernel_size=default_ks(),
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(22),
              RANGE_D=default_d()
         )],
        # k_cut
        [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='4', 
              kernel_size=default_ks(),
              padding="same",
              EDGE=1,
              RANGE_ST=default_noise_st(24),
              RANGE_D=default_d(),
              CONTRIB=1
         )],
    ])
    return kinn_model_space


def get_uniform_ms2(n_states, st_win_size=None, verbose=False):
    """an evenly-spaced model space, separating 20nt for given n_states
    """
    seqlen = 30
    if st_win_size is None:
        st_win_size = int(np.ceil(seqlen / (n_states-1)))
        if verbose: print("win size", st_win_size)
    st_win = np.arange(st_win_size) - st_win_size//2
    start_offset = 5 # could be 3- PAM
    anchors = {s:i-int(np.ceil(st_win_size/2)) for s,i in enumerate(start_offset+np.arange(0, seqlen+st_win_size, st_win_size, dtype='int'))}
    if verbose: print("anchors", anchors)
    if verbose: print("st_win", st_win)
    ls = []
    default_ks = lambda: pmbga.Categorical(choices=[1,3,7], prior_cnt=1)
    #default_d = lambda: pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1)
    default_d = lambda: pmbga.Categorical(choices=np.arange(5, max(10,st_win_size)), prior_cnt=1)
    default_st = lambda a: pmbga.Categorical(choices=np.clip(a+st_win, 0, seqlen), prior_cnt=1)
    for s in range(0, n_states-1):
        if verbose: print(s, default_st(anchors[s]).choice_lookup)
        ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s), TARGET=str(s+1),
            EDGE=1,
            kernel_size=default_ks(),
            padding="same",
            RANGE_ST=default_st(anchors[s]),
            RANGE_D=default_d(),
            CONTRIB=int(s==(n_states-1))
            )])

        ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s+1), TARGET=str(s), EDGE=1,
            kernel_size=default_ks(),
            padding="same",
            RANGE_ST=default_st(anchors[s]),
            RANGE_D=default_d()
            )])
    # last rate: cleavage, irreversible
    return ModelSpace.from_dict(ls)


def get_cas9_finkelstein_ms(use_sink_state=False):
    """model space based on https://www.biorxiv.org/content/10.1101/2020.05.21.108613v2
    """
    ks_choices=[1,3,7]
    kinn_model_space = ModelSpace.from_dict([
        # k_on, sol -> open R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='0', TARGET='1',
              kernel_size=3,
              padding="valid",
              EDGE=1,
              RANGE_ST=0,
              RANGE_D=3,
         )],
        # k_off, open R-loop -> sol
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='0', 
              kernel_size=3,
              padding="valid",
              EDGE=1,
              RANGE_ST=0,
              RANGE_D=3
         )],
        # k_OI, open R-loop -> intermediate R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='2', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=1), 
              padding="same",
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[3,4,5,6,7,8,9,10,11,12], 
                  prior_cnt=1),
              RANGE_D=pmbga.Categorical(choices=[5,6,7,8,9,10], prior_cnt=1) 
         )],
        # k_IO, intermediate R-loop -> open R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='1', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=1), 
              padding="same",
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[3,4,5,6,7,8,9,10,11,12],
                  prior_cnt=1),
              RANGE_D=pmbga.Categorical(choices=[5,6,7,8,9,10], prior_cnt=1) 
         )],
        # k_IC, intermediate R-loop -> closed R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='2', TARGET='3', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=1), 
              padding="same",
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=[11,12,13,14,15,16,17], prior_cnt=1),
              RANGE_D=pmbga.Categorical(choices=[5,6,7,8,9,10], prior_cnt=1) 
         )],
        # k_CI, closed R-loop -> intermediate R-loop
        [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='2', 
              kernel_size=pmbga.Categorical(choices=ks_choices, prior_cnt=1),
              padding="same",
              EDGE=1,        
              RANGE_ST=pmbga.Categorical(choices=[11,12,13,14,15,16,17], prior_cnt=1),
              RANGE_D=pmbga.Categorical(choices=[5,6,7,8,9,10], prior_cnt=1) 
         )],
        # k_30 for cycle, k_34 for sink
        [dict(Layer_type='conv1d', filters=1, SOURCE='3', TARGET='4' if use_sink_state else '0', 
              kernel_size=pmbga.Categorical(choices=[1,3,5,7], prior_cnt=1),
              padding="same",
              EDGE=1,
              RANGE_ST=pmbga.Categorical(choices=np.arange(0,23-5), prior_cnt=1),
              RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1), 
              #RANGE_D=pmbga.Categorical(choices=np.arange(7,15), prior_cnt=1),
              CONTRIB=1
         )],
    ])
    return kinn_model_space


def get_cas9_uniform_ms(n_states, st_win_size=None, verbose=False):
    """an evenly-spaced model space, separating 20nt for given n_states
    """
    if st_win_size is None:
        st_win_size = int(np.ceil(20 / (n_states-2)))
        print("win size", st_win_size)
    st_win = np.arange(st_win_size) - st_win_size//2
    anchors = {s:i-int(np.ceil(st_win_size/2)) for s,i in enumerate(3+np.arange(0, 20+st_win_size, st_win_size, dtype='int'))}
    print("anchors", anchors)
    print("st_win", st_win)
    ls = []
    default_ks = lambda: pmbga.Categorical(choices=[1,3,7], prior_cnt=1)
    #default_d = lambda: pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1)
    default_d = lambda: pmbga.Categorical(choices=np.arange(5, max(10,st_win_size)), prior_cnt=1)
    default_st = lambda a: pmbga.Categorical(choices=np.clip(a+st_win, 0, 23), prior_cnt=1)
    # sol -> open R loop is fixed
    ls.extend([
        [dict(Layer_type='conv1d', filters=1, SOURCE='0', TARGET='1', EDGE=1,
            kernel_size=3,
            padding="valid",
            RANGE_ST=0,
            RANGE_D=3
            )],
        [dict(Layer_type='conv1d', filters=1, SOURCE='1', TARGET='0', EDGE=1,
            kernel_size=3,
            padding="valid",
            RANGE_ST=0,
            RANGE_D=3
            )],
    ])
    for s in range(1, n_states-1):
        print(s, default_st(anchors[s]).choice_lookup)
        ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s), TARGET=str(s+1), EDGE=1,
            kernel_size=default_ks(),
            padding="same",
            RANGE_ST=default_st(anchors[s]),
            RANGE_D=default_d()
            )])
        ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s+1), TARGET=str(s), EDGE=1,
            kernel_size=default_ks(),
            padding="same",
            RANGE_ST=default_st(anchors[s]),
            RANGE_D=default_d()
            )])
    # last rate: cleavage, irreversible
    if n_states==2: s=0
    ls.append([dict(Layer_type='conv1d', filters=1, SOURCE=str(s+1), TARGET='0', EDGE=1,
            kernel_size=default_ks(),
            padding="same",
            RANGE_ST=pmbga.Categorical(choices=np.arange(0, 20), prior_cnt=1),
            RANGE_D=pmbga.ZeroTruncatedNegativeBinomial(alpha=5, beta=1),
            #RANGE_D=pmbga.Categorical(choices=np.arange(7,15), prior_cnt=1),
            CONTRIB=1
            )])
    return ModelSpace.from_dict(ls)


