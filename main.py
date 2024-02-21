import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import read_data
import model_general
import post_processing
import performance

CNN_LSTM = True
CNN = False
TRAINING = True
COMPARISON = False
MITDB = True
chunk = 1000
train_size = 84

df_merged = pd.read_hdf('df_merged.h5', 'df')

xTrain, xTest, df_train, df_test = read_data.split_data(train_size, df_merged)

xTrain[xTrain > 200] = 200
xTrain[xTrain < -200] = -200
xTest[xTest > 200] = 200
xTest[xTest < -200] = -200

xTrain_scaled, xTest_scaled = read_data.scale_data(xTrain, xTest)

##################

df_train['annotation'] = np.where(df_train['annotation'] == 5, 1, df_train['annotation'])
df_test['annotation'] = np.where(df_test['annotation'] == 5, 1, df_test['annotation'])
yTrain_class = to_categorical(df_train['annotation'].values-1)
yTest_class = to_categorical(df_test['annotation'].values-1)

yTrain_class = np.array(read_data.create_subseq(yTrain_class, chunk))
yTest_class = np.array(read_data.create_subseq(yTest_class, chunk))

print('Shape of train targets: {}'.format(yTrain_class.shape))
print('Shape of test targets: {}'.format(yTest_class.shape))

####################


if not TRAINING:
    name_tr = 'Pred_Tr_84TR_bandpass_scaled'
    Pred_Tr = model_general.load_data(name_tr)

    name_ts = 'Pred_Ts_84TR_bandpass_scaled'
    Pred_Ts = model_general.load_data(name_ts)

    print('######## Train set performance ########')
    print(classification_report(yTrain_class.argmax(axis=-1).reshape((yTrain_class.shape[0] * yTrain_class.shape[1])),
                                Pred_Tr.argmax(axis=-1).reshape((Pred_Tr.shape[0] * Pred_Tr.shape[1]))))
    print('######## Test set performance ########')
    print(classification_report(yTest_class.argmax(axis=-1).reshape((yTest_class.shape[0] * yTest_class.shape[1])),
                                Pred_Ts.argmax(axis=-1).reshape((Pred_Ts.shape[0] * Pred_Ts.shape[1]))))

    df_filled_ts = post_processing.create_df_for_comparison(yTest_class, Pred_Ts)
    fn_tp = post_processing.fn_tp_compute(df_filled_ts)
    fp_tp = post_processing.fp_tp_compute(df_filled_ts)
    perf_metrics = performance.metrics_evaluation(fn_tp, fp_tp)


if TRAINING:
    if CNN_LSTM:
        xTrain_chunked = np.array(read_data.create_subseq(xTrain_scaled, chunk))
        xTrain_chunked = xTrain_chunked.reshape((xTrain_chunked.shape[0], xTrain_chunked.shape[1], 1))
        xTest_chunked = np.array(read_data.create_subseq(xTest_scaled, chunk))
        xTest_chunked = xTest_chunked.reshape((xTest_chunked.shape[0], xTest_chunked.shape[1], 1))
        print('Shape of train data: {}'.format(xTrain_chunked.shape))
        print('Shape of test data: {}'.format(xTest_chunked.shape))

        model = model_general.cnn_lstm(xTrain_chunked)
        history = model_general.model_fit_cnn_lstm(model, xTrain_chunked, yTrain_class)

    if CNN:
        xTrain_chunked = np.array(read_data.create_subseq(xTrain_scaled, chunk))
        xTrain_chunked = xTrain_chunked.reshape((xTrain_chunked.shape[0], xTrain_chunked.shape[1], 1))
        xTest_chunked = np.array(read_data.create_subseq(xTest_scaled, chunk))
        xTest_chunked = xTest_chunked.reshape((xTest_chunked.shape[0], xTest_chunked.shape[1], 1))
        print('Shape of train data: {}'.format(xTrain_chunked.shape))
        print('Shape of test data: {}'.format(xTest_chunked.shape))

        model = model_general.cnn(xTrain_chunked)
        history = model_general.model_fit_cnn(model, xTrain_chunked, yTrain_class)

    name_hist = 'model_84TR_bandpass_scaled'
    model_general.save_history(history, name_hist)
    model_name = 'model_84TR_bandpass_scaled.h5'
    model_general.save_model(model, model_name)

    Pred_Tr = model.predict(xTrain_chunked)
    Pred_Ts = model.predict(xTest_chunked)

    name_pred_Tr = 'Pred_Tr_84TR_bandpass_scaled.npz'
    model_general.save_pred(Pred_Tr, name_pred_Tr)
    name_pred_Ts = 'Pred_Ts_84TR_bandpass_scaled.npz'
    model_general.save_pred(Pred_Ts, name_pred_Ts)

    print('######## Train set performance ########')
    print(classification_report(yTrain_class.argmax(axis=-1).reshape((yTrain_class.shape[0] * yTrain_class.shape[1])),
                                Pred_Tr.argmax(axis=-1).reshape((Pred_Tr.shape[0] * Pred_Tr.shape[1]))))
    print('######## Test set performance ########')
    print(classification_report(yTest_class.argmax(axis=-1).reshape((yTest_class.shape[0] * yTest_class.shape[1])),
                                Pred_Ts.argmax(axis=-1).reshape((Pred_Ts.shape[0] * Pred_Ts.shape[1]))))

    df_filled_ts = post_processing.create_df_for_comparison(yTest_class, Pred_Ts)
    fn_tp = post_processing.fn_tp_compute(df_filled_ts)
    fp_tp = post_processing.fp_tp_compute(df_filled_ts)
    perf_metrics = performance.metrics_evaluation(fn_tp, fp_tp)
    print('qtdb completed')


if MITDB:
    df_mitdb = pd.read_csv('df_mitdb.csv', sep='\t')
    df_mitdb.drop(['Unnamed: 0'], axis=1, inplace=True)
    y_filt = read_data.butter_bandpass_filter(df_mitdb.ch_1, 0.5, 40, 360, order=3)
    df_mitdb['filt'] = y_filt
    xTest_mitdb = df_mitdb['filt'].values

    xTest_mitdb[xTest_mitdb > 2] = 2
    xTest_mitdb[xTest_mitdb < -2] = -2
    xTest_mitdb_scaled, xTest_mitdb_scaled = read_data.scale_data(xTest_mitdb, xTest_mitdb)

    yTest_class_mitdb = to_categorical(df_mitdb['symbol'].values)
    yTest_class_mitdb = np.array(read_data.create_subseq(yTest_class_mitdb, chunk))
    xTest_chunked_mitdb = np.array(read_data.create_subseq(xTest_mitdb_scaled, chunk))
    xTest_chunked_mitdb = xTest_chunked_mitdb.reshape((xTest_chunked_mitdb.shape[0], xTest_chunked_mitdb.shape[1], 1))
    #############

    model_name = 'model_84TR_bandpass_scaled.h5'
    model = model_general.model_load(model_name)
    Pred_Ts_mitdb = model.predict(xTest_chunked_mitdb)

    name_pred_Ts_mitdb = 'Pred_Ts_mitdb_bandpass_scaled.npz'
    model_general.save_pred(Pred_Ts_mitdb, name_pred_Ts_mitdb)
    #############
    df_filled_ts_mitdb = post_processing.create_df_for_comparison(yTest_class_mitdb, Pred_Ts_mitdb)
    fp_tp_mitdb = post_processing.tp_fp_mitdb(df_filled_ts_mitdb)
    fn_tp_mitdb = post_processing.tp_fn_mitdb(df_filled_ts_mitdb)
    perf_metrics_mitdb = performance.metrics_evaluation_mitdb(fn_tp_mitdb, fp_tp_mitdb)
    print('mitdb completed')


if COMPARISON:
    # Prepare test set for the manual annotations
    df_test_man = df_test.loc[df_test['man_annot'].notnull()]
    xTest_man = df_test_man['filt'].values
    yTest_class_man = to_categorical(df_test_man['man_annot'].values - 1)
    yTest_class_man = np.array(read_data.create_subseq(yTest_class_man, chunk))
    xTest_chunked_man = np.array(read_data.create_subseq(xTest_man, chunk))
    xTest_chunked_man = xTest_chunked_man.reshape((xTest_chunked_man.shape[0], xTest_chunked_man.shape[1], 1))
    # Load the model and do prediction
    model = model_general.model_load('model_84TR_bandpass.h5')
    Pred_Ts_man = model.predict(xTest_chunked_man)
    Pred_Ts_man_numerical = Pred_Ts_man.argmax(axis=-1).reshape((Pred_Ts_man.shape[0] * Pred_Ts_man.shape[1]))
    yTest_class_man_numerical = yTest_class_man.argmax(axis=-1).reshape((yTest_class_man.shape[0] * yTest_class_man.shape[1]))

    print('######## Test set performance on manual annotation ########')
    print(classification_report(yTest_class_man_numerical, Pred_Ts_man_numerical))

    df_filled, pred_filled_man = post_processing.cavity_filling(Pred_Ts_man_numerical, 10)
    df_test_man.reset_index(inplace=True)
    df_test_man_filled = pd.concat([df_test_man, df_filled], axis=1)
    df_test_man_filled['pred_filled'] = df_test_man_filled['pred_filled'] + 1
