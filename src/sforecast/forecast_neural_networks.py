from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, concatenate, Dropout
from keras.layers import LSTM, GRU
#from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd


def get_dense_nn(Nlags:int, 
                    Ncovars: int = 1, 
                    Nexogs:int = 0,
                    Nendogs:int = 0
                    ):
    
    assert Ncovars >= 1 , f'Ncovars < 0 not allowed, Ncovars = {Ncovars}'
    assert Nexogs >= 0 , f'Nexogs < 0 not allowed, Ncovars = {Nexogs}'
    assert Nendogs >= 0 , f'Nendogs < 0 not allowed, Ncovars = {Nendogs}'
    
    Ndense = Nlags * Ncovars + Nexogs + Nendogs
    
    Nin = Ndense
    Ndense = np.rint(np.max([Ndense,20])).astype(int)
    nn_model_dense = None
    Nout = Ncovars
    
    print(f'Ndense = {Nin}')
    print(f'Ndense = {Ndense}')
    print(f'Nout = {Nout}')

    # Dense Network, 2 hidden layers, continuous inputs
    inputs = Input((Nin,))
    h1c = Dense(Ndense, activation='relu')(inputs)

    # dense reduction layers
    h1c_d = Dropout(0.2)(h1c)
    Nh2c = np.rint(Ndense/2).astype(int) #round init
    h2c = Dense(Nh2c, activation='relu')(h1c_d)
    h2c_d = Dropout(0.2)(h2c)

    # output
    output = Dense(Nout)(h2c_d)  # linear activation ... linear combination 
    nn_model_dense = Model(inputs=inputs,outputs=output)

    # define optimizer and compile ...
    # decay warning
    # https://stackoverflow.com/questions/74734685/how-to-fix-this-value-error-valueerror-decay-is-deprecated-in-the-new-keras-o
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    nn_model_dense.compile(loss='mse', optimizer=optimizer)

    print(nn_model_dense.summary())

    return nn_model_dense

def get_dense_emb_nn(df: pd.DataFrame,
                    Nlags:int, 
                    catvars: list[str],
                    Ncovars: int = 1, 
                    Nexogs:int = 0,
                    Nendogs:int = 0):
    

    
    assert Ncovars >= 1 , f'Ncovars < 0 not allowed, Ncovars = {Ncovars}'
    assert Nexogs >= 0 , f'Nexogs < 0 not allowed, Ncovars = {Nexogs}'
    assert Nendogs >= 0 , f'Nendogs < 0 not allowed, Ncovars = {Nendogs}'
    
    Ndense = Nlags * Ncovars +  Nexogs + Nendogs
    nn_model_dense_emb  = None
    
    Nemb = len(catvars)
    eindim = [df[catvars].groupby(c)[c].count().index.size + 1 for c in catvars] # add 1 to the dim or err in TF
    eoutdim = [np.rint(np.log2(x)).astype(int) for x in eindim]

    Nembout = sum(eoutdim)

    # Nout
    Nout = Ncovars
    
    # dense inputs
    Ndin = Ndense
    Ndense = np.rint(np.max([Ndense,20])).astype(int)

    print(f'Ndin = {Ndin}')
    print(f'Ndense = {Ndense}')
    print(f'Nemb = {Nemb}')
    print(f'Nout = {Nout}')


    # Dense Network, 2 hidden layers, continuous variables ... covar lags and exogenous variables
    cont_inputs = Input((Ndin,))
    h1d = Dense(Ndense, activation='relu')(cont_inputs)

    # embeddings, cat vars
    cat_inputs_list = [ Input((1,)) for c in range(Nemb) ]  # one embedding for each categorical variable
    emb_out_list = [Embedding(ein,eout,input_length=1)(cat) for ein,eout,cat in zip(eindim ,eoutdim,cat_inputs_list) ]
    emb_flat_list = [Flatten()(emb_out) for emb_out in emb_out_list ]

    # combined 
    combined = concatenate([h1d]+emb_flat_list)
    combined_d = Dropout(0.2)(combined)

    # dense reduction layers
    Nh1c = Ndense + Nembout # 
    h1c = Dense(Nh1c, activation='relu')(combined_d)
    h1c_d = Dropout(0.2)(h1c)
    Nh2c = np.rint(Nh1c/2).astype(int)
    h2c = Dense(Nh2c, activation='relu')(h1c_d)
    h2c_d = Dropout(0.2)(h2c)

    # output

    print(f'cont_inputs = {cont_inputs}')
    print(f'cat_inputs_list = {cat_inputs_list}')

    output = Dense(Nout)(h2c_d)  # linear activation ... linear combination 
    nn_model_dense_emb = Model(inputs=[cont_inputs] + cat_inputs_list, outputs=output)

    # define optimizer and compile ...
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    nn_model_dense_emb.compile(loss='mse', optimizer=optimizer)

    print(nn_model_dense_emb.summary())
    
    return nn_model_dense_emb