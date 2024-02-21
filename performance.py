import pandas as pd


def metrics_evaluation(fn_tp, fp_tp):
    sen_peak_p = fn_tp['tp_peak_p']/(fn_tp['tp_peak_p']+fn_tp['fn_peak_p'])
    sen_peak_r = fn_tp['tp_peak_r']/(fn_tp['tp_peak_r']+fn_tp['fn_peak_r'])
    sen_peak_t = fn_tp['tp_peak_t']/(fn_tp['tp_peak_t']+fn_tp['fn_peak_t'])

    sen_st_p = fn_tp['tp_st_p'] / (fn_tp['tp_st_p'] + fn_tp['fn_st_p'])
    sen_st_r = fn_tp['tp_st_r'] / (fn_tp['tp_st_r'] + fn_tp['fn_st_r'])
    sen_st_t = fn_tp['tp_st_t'] / (fn_tp['tp_st_t'] + fn_tp['fn_st_t'])

    sen_end_p = fn_tp['tp_end_p'] / (fn_tp['tp_end_p'] + fn_tp['fn_end_p'])
    sen_end_r = fn_tp['tp_end_r'] / (fn_tp['tp_end_r'] + fn_tp['fn_end_r'])
    sen_end_t = fn_tp['tp_end_t'] / (fn_tp['tp_end_t'] + fn_tp['fn_end_t'])

    ppv_peak_p = fp_tp['tp_peak_p'] / (fp_tp['tp_peak_p'] + fp_tp['fp_peak_p'])
    ppv_peak_r = fp_tp['tp_peak_r'] / (fp_tp['tp_peak_r'] + fp_tp['fp_peak_r'])
    ppv_peak_t = fp_tp['tp_peak_t'] / (fp_tp['tp_peak_t'] + fp_tp['fp_peak_t'])

    ppv_st_p = fp_tp['tp_st_p'] / (fp_tp['tp_st_p'] + fp_tp['fp_st_p'])
    ppv_st_r = fp_tp['tp_st_r'] / (fp_tp['tp_st_r'] + fp_tp['fp_st_r'])
    ppv_st_t = fp_tp['tp_st_t'] / (fp_tp['tp_st_t'] + fp_tp['fp_st_t'])

    ppv_end_p = fp_tp['tp_end_p'] / (fp_tp['tp_end_p'] + fp_tp['fp_end_p'])
    ppv_end_r = fp_tp['tp_end_r'] / (fp_tp['tp_end_r'] + fp_tp['fp_end_r'])
    ppv_end_t = fp_tp['tp_end_t'] / (fp_tp['tp_end_t'] + fp_tp['fp_end_t'])

    f1_peak_p = 2*(sen_peak_p * ppv_peak_p)/(sen_peak_p + ppv_peak_p)
    f1_peak_r = 2*(sen_peak_r * ppv_peak_r)/(sen_peak_r + ppv_peak_r)
    f1_peak_t = 2*(sen_peak_t * ppv_peak_t)/(sen_peak_t + ppv_peak_t)

    f1_st_p = 2 * (sen_st_p * ppv_st_p) / (sen_st_p + ppv_st_p)
    f1_st_r = 2 * (sen_st_r * ppv_st_r) / (sen_st_r + ppv_st_r)
    f1_st_t = 2 * (sen_st_t * ppv_st_t) / (sen_st_t + ppv_st_t)

    f1_end_p = 2 * (sen_end_p * ppv_end_p) / (sen_end_p + ppv_end_p)
    f1_end_r = 2 * (sen_end_r * ppv_end_r) / (sen_end_r + ppv_end_r)
    f1_end_t = 2 * (sen_end_t * ppv_end_t) / (sen_end_t + ppv_end_t)

    results = {'sen_peak_p': round(sen_peak_p*100, 2), 'sen_peak_r': round(sen_peak_r*100, 2), 'sen_peak_t': round(sen_peak_t*100, 2),
               'sen_st_p': round(sen_st_p*100, 2), 'sen_st_r': round(sen_st_r*100, 2), 'sen_st_t': round(sen_st_t*100, 2),
               'sen_end_p': round(sen_end_p*100, 2), 'sen_end_r': round(sen_end_r*100, 2), 'sen_end_t': round(sen_end_t*100, 2),
               'ppv_peak_p': round(ppv_peak_p*100, 2), 'ppv_peak_r': round(ppv_peak_r*100, 2), 'ppv_peak_t': round(ppv_peak_t*100, 2),
               'ppv_st_p': round(ppv_st_p*100, 2), 'ppv_st_r': round(ppv_st_r*100, 2), 'ppv_st_t': round(ppv_st_t*100, 2),
               'ppv_end_p': round(ppv_end_p*100, 2), 'ppv_end_r': round(ppv_end_r*100, 2), 'ppv_end_t': round(ppv_end_t*100, 2),
               'f1_peak_p': round(f1_peak_p*100, 2), 'f1_peak_r': round(f1_peak_r*100, 2), 'f1_peak_t': round(f1_peak_t*100, 2),
               'f1_st_p': round(f1_st_p*100, 2), 'f1_st_r': round(f1_st_r*100, 2), 'f1_st_t': round(f1_st_t*100, 2),
               'f1_end_p': round(f1_end_p*100, 2), 'f1_end_r': round(f1_end_r*100, 2), 'f1_end_t': round(f1_end_t*100, 2),
               }
    df_sen_spe_f1 = pd.DataFrame(list(results.items()), columns=['measure', 'rate'])
    df_sen_spe_f1.to_csv('df_sen_spe_f1_qtdb.csv', sep='\t')
    return results


def metrics_evaluation_mitdb(fn_tp, fp_tp):
    sen_peak_r = fn_tp['tp_peak_r'] / (fn_tp['tp_peak_r'] + fn_tp['fn_peak_r'])
    ppv_peak_r = fp_tp['tp_peak_r'] / (fp_tp['tp_peak_r'] + fp_tp['fp_peak_r'])
    f1_peak_r = 2 * (sen_peak_r * ppv_peak_r) / (sen_peak_r + ppv_peak_r)

    results = {'sen_peak_r': round(sen_peak_r * 100, 2), 'ppv_peak_r': round(ppv_peak_r * 100, 2),
               'f1_peak_r': round(f1_peak_r * 100, 2)}
    df_sen_spe_f1_mitdb = pd.DataFrame(list(results.items()), columns=['measure', 'rate'])
    df_sen_spe_f1_mitdb.to_csv('df_sen_spe_f1_mitdb.csv', sep='\t')
    return results

