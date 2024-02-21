import scipy.io
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from scipy.signal import butter, sosfilt, sosfreqz
#os.chdir("..")
DATA_PATH = os.path.dirname(os.getcwd())
OUTPUT_DIR = r"data"


def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)
    return y


def load_data():
    annot = scipy.io.loadmat('values_ed.mat')
    signal = scipy.io.loadmat('source_ed.mat')
    annot_first = []
    for i in range(0, annot['Values'].shape[1], 2):
        annot_first.append(annot['Values'][:, i])
    annot_first_np = np.asarray(annot_first)
    annot_first_1d = annot_first_np.ravel()
    arr = np.arange(1, 106)
    idx = np.repeat(arr, 225000)
    samp = np.array(list(np.arange(0, 225000)) * 105)
    sig = signal['Source'][:, 0][:]
    sig_1d = sig.ravel()
    df_all = pd.DataFrame({'samples': samp, 'ch_1': sig_1d, 'annotation': annot_first_1d, 'id': idx},
                          columns=['samples', 'ch_1', 'annotation', 'id'])
    ##############
    df_man = pd.read_hdf('df_man.h5', 'df')
    df_man['id'] = df_man['id'] + 1
    df_man = df_man.rename(columns={"annotation": "man_annot"}, errors="raise")
    df_man = df_man.drop(['ch_1'], axis=1)
    ##############
    df_all_fill = df_all.fillna(0)
    # Filtered signal and shifting to match the original annotation
    y_filt = butter_bandpass_filter(df_all_fill.ch_1, 0.5, 40, 250, order=3)
    df_all_fill['filt'] = y_filt
    df_merged = pd.merge(df_all_fill, df_man, how='left', on=['samples', 'id'])
    return df_all_fill, df_merged #df_all_shift


def split_data(train_size, df_all_shift):
    seed = 1234
    tr_idx = random.Random(seed).sample(range(0, 105), train_size)
    df_train = df_all_shift.loc[df_all_shift['id'].isin(tr_idx)]
    df_test = df_all_shift.loc[~df_all_shift['id'].isin(tr_idx)]
    xTrain = df_train['ch_1'].values
    xTest = df_test['ch_1'].values
    return xTrain, xTest, df_train, df_test


def scale_data(xTrain, xTest):
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(xTrain.reshape(-1, 1))
    xTrain_scaled = scaler.transform(xTrain.reshape(-1, 1))
    xTest_scaled = scaler.transform(xTest.reshape(-1, 1))
    return xTrain_scaled, xTest_scaled


def create_subseq(ts, look_back):
    sub_seq = []
    for ii in range(int(len(ts) / look_back)):
        sub_seq.append(ts[ii * look_back:(ii + 1) * look_back])
    return sub_seq


def save_x_h5(filename, X):
    return X.to_hdf(os.path.join(DATA_PATH, OUTPUT_DIR, f"{filename}.h5"), "X", mode="w", complevel=9, complib="zlib")


def read_x_h5(filename):
    return pd.read_hdf(os.path.join(DATA_PATH, OUTPUT_DIR, f"{filename}.h5"), "X", mode="r")
