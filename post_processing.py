import pandas as pd
import numpy as np


def cavity_filling(pred, win_size):
    for i in range(int(len(pred))-win_size):
        if pred[i] == pred[i+win_size]:
            pred[i:i+win_size] = [pred[i]]*win_size
        elif pred[i] != pred[i+win_size]:
            pass
    return pd.DataFrame({'pred_filled': pred}), pred


def create_df_for_comparison(actual, pred):
    actual_numerical = actual.argmax(axis=-1).reshape((actual.shape[0] * actual.shape[1]))
    pred_numerical = pred.argmax(axis=-1).reshape((pred.shape[0] * pred.shape[1]))
    df_temp, pred_filled = cavity_filling(pred_numerical, 10)

    df = pd.DataFrame({'actual': actual_numerical, 'pred': pred_numerical, 'pred_filled': pred_filled})
    return df


def fn_tp_compute(df):
    tp_peak1 = 0
    tp_peak2 = 0
    tp_peak3 = 0
    tp_st1 = 0
    tp_st2 = 0
    tp_st3 = 0
    tp_end1 = 0
    tp_end2 = 0
    tp_end3 = 0
    fn_peak1 = 0
    fn_peak2 = 0
    fn_peak3 = 0
    fn_st1 = 0
    fn_st2 = 0
    fn_st3 = 0
    fn_end1 = 0
    fn_end2 = 0
    fn_end3 = 0
    st = 0
    for i in range(47, df.index[:-1].shape[0]): # df.index[:-1]:
        if df.loc[i, 'actual'] != 0: # check if there is a class
            if df.loc[i-1, 'actual'] != df.loc[i, 'actual']: # start of the class
                st = i
                if df.loc[i, 'actual'] == 1:
                    p_thresh = 11
                    df_temp = df.iloc[i-p_thresh:i+p_thresh+1] # look at the threshold boundary
                    first = df_temp[df_temp['pred_filled'] == 1].index.min() # get the location of the first predicted class 1 in the boundary
                    if first == i-p_thresh: # check if the first element of this boundary is equla to the index of i-p_thresh
                        if df.loc[i-p_thresh-1, 'pred_filled'] == 1:
                            fn_st1 += 1
                        else:
                            tp_st1 += 1
                    elif np.isnan(first): # if there is no class 1 in the boundary
                        fn_st1 += 1
                    else:
                        tp_st1 += 1

                elif df.loc[i, 'actual'] == 2:
                    r_thresh = 7
                    df_temp = df.iloc[i-r_thresh:i+r_thresh+1]
                    first = df_temp[df_temp['pred_filled'] == 2].index.min()
                    if first == i - r_thresh:
                        if df.loc[i - r_thresh - 1, 'pred_filled'] == 2:
                            fn_st2 += 1
                        else:
                            tp_st2 += 1
                    elif np.isnan(first):
                        fn_st2 += 1
                    else:
                        tp_st2 += 1

                elif df.loc[i, 'actual'] == 3:
                    t_thresh = 30
                    df_temp = df.iloc[i - t_thresh:i + t_thresh + 1]
                    first = df_temp[df_temp['pred_filled'] == 3].index.min()
                    if first == i - t_thresh:
                        if df.loc[i - t_thresh - 1, 'pred_filled'] == 3:
                            fn_st3 += 1
                        else:
                            tp_st3 += 1
                    elif np.isnan(first):
                        fn_st3 += 1
                    else:
                        tp_st3 += 1

            if df.loc[i+1, 'actual'] != df.loc[i, 'actual']:
                if df.loc[i, 'actual'] == 1:
                    print(i)
                    p_thresh = 13
                    df_temp = df.iloc[i - p_thresh:i + p_thresh + 1]
                    last = df_temp[df_temp['pred_filled'] == 1].index.max()
                    if last == i + p_thresh:
                        if df.loc[i + p_thresh + 1, 'pred_filled'] == 1:
                            fn_end1 += 1
                        else:
                            tp_end1 += 1
                    elif np.isnan(last):
                        fn_end1 += 1
                    else:
                        tp_end1 += 1
                    if 1 in df.iloc[st:i+1, 2].values:
                        tp_peak1 += 1
                    else:
                        fn_peak1 += 1
                elif df.loc[i, 'actual'] == 2:
                    r_thresh = 12
                    df_temp = df.iloc[i - r_thresh:i + r_thresh + 1]
                    last = df_temp[df_temp['pred_filled'] == 2].index.max()
                    if last == i + r_thresh:
                        if df.loc[i + r_thresh + 1, 'pred_filled'] == 2:
                            fn_end2 += 1
                        else:
                            tp_end2 += 1
                    elif np.isnan(last):
                        fn_end2 += 1
                    else:
                        tp_end2 += 1
                    if 2 in df.iloc[st:i+1, 2].values:
                        tp_peak2 += 1
                    else:
                        fn_peak2 += 1

                elif df.loc[i, 'actual'] == 3:
                    t_thresh = 31
                    df_temp = df.iloc[i - t_thresh:i + t_thresh + 1]
                    last = df_temp[df_temp['pred_filled'] == 3].index.max()
                    if last == i + t_thresh:
                        if df.loc[i + t_thresh + 1, 'pred_filled'] == 3:
                            fn_end3 += 1
                        else:
                            tp_end3 += 1
                    elif np.isnan(last):
                        fn_end3 += 1
                    else:
                        tp_end3 += 1
                    if 3 in df.iloc[st:i+1, 2].values:
                        tp_peak3 += 1
                    else:
                        fn_peak3 += 1
    results = {'tp_peak_p': tp_peak1, 'tp_peak_r': tp_peak2, 'tp_peak_t': tp_peak3,
                'tp_st_p': tp_st1, 'tp_st_r': tp_st2, 'tp_st_t': tp_st3,
                'tp_end_p': tp_end1, 'tp_end_r': tp_end2, 'tp_end_t': tp_end3,
                'fn_peak_p': fn_peak1, 'fn_peak_r': fn_peak2, 'fn_peak_t': fn_peak3,
                'fn_st_p': fn_st1, 'fn_st_r': fn_st2, 'fn_st_t': fn_st3,
                'fn_end_p': fn_end1, 'fn_end_r': fn_end2, 'fn_end_t': fn_end3}
    df_fn_tp = pd.DataFrame(list(results.items()), columns=['measure', 'rate'])
    df_fn_tp.to_csv('df_fn_tp_qtdb.csv', sep='\t')
    return results


def fp_tp_compute(df):
    tp_peak1 = 0
    tp_peak2 = 0
    tp_peak3 = 0
    tp_st1 = 0
    tp_st2 = 0
    tp_st3 = 0
    tp_end1 = 0
    tp_end2 = 0
    tp_end3 = 0
    fp_peak1 = 0
    fp_peak2 = 0
    fp_peak3 = 0
    fp_st1 = 0
    fp_st2 = 0
    fp_st3 = 0
    fp_end1 = 0
    fp_end2 = 0
    fp_end3 = 0
    st = 0
    for i in range(47, df.index[:-1].shape[0]): # df.index[:-1]:
        if df.loc[i, 'pred_filled'] != 0: # check if there is a class
            if df.loc[i-1, 'pred_filled'] != df.loc[i, 'pred_filled']: # start of the class
                st = i
                if df.loc[i, 'pred_filled'] == 1:
                    p_thresh = 11
                    df_temp = df.iloc[i-p_thresh:i+p_thresh+1] # look at the threshold boundary
                    first = df_temp[df_temp['actual'] == 1].index.min() # get the location of the first predicted class 1 in the boundary
                    if first == i-p_thresh: # check if the first element of this boundary is equla to the index of i-p_thresh
                        if df.loc[i-p_thresh-1, 'actual'] == 1:
                            fp_st1 += 1
                        else:
                            tp_st1 += 1
                    elif np.isnan(first): # if there is no class 1 in the boundary
                        fp_st1 += 1
                    else:
                        tp_st1 += 1

                elif df.loc[i, 'pred_filled'] == 2:
                    r_thresh = 7
                    df_temp = df.iloc[i-r_thresh:i+r_thresh+1]
                    first = df_temp[df_temp['actual'] == 2].index.min()
                    if first == i - r_thresh:
                        if df.loc[i - r_thresh - 1, 'actual'] == 2:
                            fp_st2 += 1
                        else:
                            tp_st2 += 1
                    elif np.isnan(first):
                        fp_st2 += 1
                    else:
                        tp_st2 += 1

                elif df.loc[i, 'pred_filled'] == 3:
                    t_thresh = 30
                    df_temp = df.iloc[i - t_thresh:i + t_thresh + 1]
                    first = df_temp[df_temp['actual'] == 3].index.min()
                    if first == i - t_thresh:
                        if df.loc[i - t_thresh - 1, 'actual'] == 3:
                            fp_st3 += 1
                        else:
                            tp_st3 += 1
                    elif np.isnan(first):
                        fp_st3 += 1
                    else:
                        tp_st3 += 1

            if df.loc[i+1, 'pred_filled'] != df.loc[i, 'pred_filled']:
                if df.loc[i, 'pred_filled'] == 1:
                    print(i)
                    p_thresh = 13
                    df_temp = df.iloc[i - p_thresh:i + p_thresh + 1]
                    last = df_temp[df_temp['actual'] == 1].index.max()
                    if last == i + p_thresh:
                        if df.loc[i + p_thresh + 1, 'actual'] == 1:
                            fp_end1 += 1
                        else:
                            tp_end1 += 1
                    elif np.isnan(last):
                        fp_end1 += 1
                    else:
                        tp_end1 += 1
                    if 1 in df.iloc[st:i+1, 0].values:
                        tp_peak1 += 1
                    else:
                        fp_peak1 += 1
                elif df.loc[i, 'pred_filled'] == 2:
                    r_thresh = 12
                    df_temp = df.iloc[i - r_thresh:i + r_thresh + 1]
                    last = df_temp[df_temp['actual'] == 2].index.max()
                    if last == i + r_thresh:
                        if df.loc[i + r_thresh + 1, 'actual'] == 2:
                            fp_end2 += 1
                        else:
                            tp_end2 += 1
                    elif np.isnan(last):
                        fp_end2 += 1
                    else:
                        tp_end2 += 1
                    if 2 in df.iloc[st:i+1, 0].values:
                        tp_peak2 += 1
                    else:
                        fp_peak2 += 1

                elif df.loc[i, 'pred_filled'] == 3:
                    t_thresh = 31
                    df_temp = df.iloc[i - t_thresh:i + t_thresh + 1]
                    last = df_temp[df_temp['actual'] == 3].index.max()
                    if last == i + t_thresh:
                        if df.loc[i + t_thresh + 1, 'actual'] == 3:
                            fp_end3 += 1
                        else:
                            tp_end3 += 1
                    elif np.isnan(last):
                        fp_end3 += 1
                    else:
                        tp_end3 += 1
                    if 3 in df.iloc[st:i+1, 0].values:
                        tp_peak3 += 1
                    else:
                        fp_peak3 += 1
    results = {'tp_peak_p': tp_peak1, 'tp_peak_r': tp_peak2, 'tp_peak_t': tp_peak3,
                'tp_st_p': tp_st1, 'tp_st_r': tp_st2, 'tp_st_t': tp_st3,
                'tp_end_p': tp_end1, 'tp_end_r': tp_end2, 'tp_end_t': tp_end3,
                'fp_peak_p': fp_peak1, 'fp_peak_r': fp_peak2, 'fp_peak_t': fp_peak3,
                'fp_st_p': fp_st1, 'fp_st_r': fp_st2, 'fp_st_t': fp_st3,
                'fp_end_p': fp_end1, 'fp_end_r': fp_end2, 'fp_end_t': fp_end3}
    df_fp_tp = pd.DataFrame(list(results.items()), columns=['measure', 'rate'])
    df_fp_tp.to_csv('df_fp_tp_qtdb.csv', sep='\t')
    return results


def tp_fp_mitdb(df):
    st = 0
    tp_peak = 0
    fp_peak = 0
    for i in df.index:
        if df.loc[i, 'pred_filled'] == 2:
            if (i > 0 and df.loc[i-1, 'pred_filled'] != 2) or i == 0:
                st = i
            if i == df.shape[0]-1 or (df.loc[i+1, 'pred_filled'] != 2):
                if 1 in df.iloc[st:i + 1, 0].values:
                    tp_peak += 1
                else:
                    fp_peak += 1

    results = {'tp_peak_r': tp_peak, 'fp_peak_r': fp_peak}
    df_fp_tp_mitdb = pd.DataFrame(list(results.items()), columns=['measure', 'rate'])
    df_fp_tp_mitdb.to_csv('df_fp_tp_mitdb.csv', sep='\t')
    return results


def tp_fn_mitdb(df):
    tp_peak = 0
    fn_peak = 0
    for i in df.index:
        if df.loc[i, 'actual'] == 1:
            if df.loc[i, 'pred_filled'] == 2:
                tp_peak += 1
            else:
                fn_peak += 1

    results = {'tp_peak_r': tp_peak, 'fn_peak_r': fn_peak}
    df_fn_tp_mitdb = pd.DataFrame(list(results.items()), columns=['measure', 'rate'])
    df_fn_tp_mitdb.to_csv('df_fn_tp_mitdb.csv', sep='\t')
    return results

