import time
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Conv1D, Conv2D
from keras.layers import LSTM, Input, Bidirectional
from keras.layers import TimeDistributed, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from tensorflow import keras
import keras


def cnn_lstm(xTrain_chunked):
    adam = optimizers.Adam(lr=0.001)

    input_layer = Input(shape=(xTrain_chunked.shape[1], xTrain_chunked.shape[2]))
    conv_1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv_1')(input_layer)
    conv_2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv_2')(conv_1)
    conv_3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', name='conv_3')(conv_2)
    bilstm_1 = Bidirectional(LSTM(250, return_sequences=True, name='bilstm_1', dropout=0.3))(conv_3, training=True)
    bilstm_2 = Bidirectional(LSTM(125, return_sequences=True, name='bilstm_2', dropout=0.3))(bilstm_1, training=True)
    pred = TimeDistributed(Dense(4, activation='softmax', name='TimeDistributed_Dense_1'))(bilstm_2)
    model = Model(inputs=input_layer, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    model.summary()

    return model


def cnn(xTrain_chunked):
    adam = optimizers.Adam(lr=0.1)

    input_layer = Input(shape=(xTrain_chunked.shape[1], xTrain_chunked.shape[2]))
    conv_1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='conv_1')(input_layer)
    conv_2 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', name='conv_2')(conv_1)
    conv_3 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv_3')(conv_2)
    pred = TimeDistributed(keras.layers.Dense(4, activation='softmax'))(conv_3)
    model = Model(inputs=input_layer, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    model.summary()

    return model


def model_fit_cnn_lstm(model, xTrain_chunked, yTrain_class):
    print('Training of the model starts ...')
    start = time.perf_counter()
    overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, mode='auto')
    history = model.fit(xTrain_chunked, yTrain_class, validation_split=0.2, epochs=50,
                        batch_size=64, callbacks=[overfitCallback])
    elapsed = time.perf_counter() - start
    print('\nTraining of the model completed in %.3f seconds.' % elapsed)
    return history


def model_fit_cnn(model, xTrain_chunked, yTrain_class):
    print('Training of the model starts ...')
    start = time.perf_counter()
    overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto')
    history = model.fit(xTrain_chunked, yTrain_class, validation_split=0.2, epochs=100,
                        batch_size=64, callbacks=[overfitCallback])
    elapsed = time.perf_counter() - start
    print('\nTraining of the model completed in %.3f seconds.' % elapsed)
    return history


def save_history(history, name):
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = str(name) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def save_model(model, model_name):
    model.save(model_name)
    print("Saved model to disk named as {}".format(model_name))


def model_predict(model, input_data):
    pred_out = model.predict(input_data)
    return pred_out


def save_pred(pred, name):
    np.savez_compressed(name, a=pred)


def load_pred(name):
    loaded_all = np.load(name)
    pred_all = loaded_all['a']
    return pred_all


def model_load(name):
    model = load_model(name)
    print("Loaded model from disk")
    return model


def load_data(data_file_name):
    loaded_all = np.load('{}.npz'.format(data_file_name))
    Pred_all = loaded_all['a']
    return Pred_all
