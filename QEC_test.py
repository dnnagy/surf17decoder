from QEC_full import print_t
from QEC_full import fit_model
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import time


def make_test(cycles, file_base, db_path):
    print_t("============================================================================")
    print_t("Fitting model with cycle length {0}".format(cycles))    
    print_t("============================================================================")

    fit_baseline=False

    model, history, (bgt, bgv) = fit_model(db_path + file_base + "_c" + str(cycles) + '_train.db', 
                               db_path + file_base + "_c" + str(cycles) + '_validation.db',
                               db_path + file_base + "_c" + str(cycles) + '_test.db', 
                               batch_size=64, 
                               early_stop=True, 
                               early_stop_min_delta=1e-7, 
                               cycle_length=cycles,
                               n_epochs=50,
                               n_workers=96,
                               baseline=fit_baseline)

    datafile_prefix = 'baseline' if fit_baseline==True else 'simpledec'
    
    # Save history to a file
    dfhist = pd.DataFrame(history.history)
    dfhist.to_csv(datafile_prefix + '_' + file_base + '_c' + str(cycles) + "_history_" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv")

    # Calculate roc_auc and save fpr, tpr as a DataFrame
    Xs = [bgv.__getitem__(k)[0] for k in range(min(len(bgv), int(np.ceil(4000/bgv.batch_size))))]
    ys = [bgv.__getitem__(k)[1] for k in range(min(len(bgv), int(np.ceil(4000/bgv.batch_size))))]
    
    Xs = np.vstack(Xs)
    ys = np.vstack(ys)
    
    y_pred = model.predict(Xs)
    fpr, tpr, thr = roc_curve(ys[:,0], y_pred[:,0])
    
    dfroc = pd.DataFrame({'fpr':np.array(fpr), 'tpr': np.array(tpr)})
    dfroc.to_csv(datafile_prefix + '_' + file_base + '_c' + str(cycles) + "_roc_" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv")
    
    print_t("AUC score={0}".format(auc(fpr, tpr)))
    print_t("{0} done.".format(file_base + "_c" + str(cycles)))
    print_t("============================================================================")
    
    return


db_path = './data/' # /content/gdrive/My Drive/deeplea2f18em/qecdata/

cycles = [100, 150, 200]
for c in cycles:
    make_test(c, "big", db_path)