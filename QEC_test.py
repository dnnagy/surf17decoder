from QEC_full import print_t
from QEC_full import fit_model
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import time

cycle_lengths = [100, 150, 200]

train_files = ['big_c{0}_train.db'.format(k) for k in cycle_lengths]
val_files = ['big_c{0}_validation.db'.format(k) for k in cycle_lengths]
test_files = ['big_c{0}_test.db'.format(k) for k in cycle_lengths]

db_path = './data/' # /content/gdrive/My Drive/deeplea2f18em/qecdata/

baseline=True

for k in range(len(cycle_lengths)):
    print_t("======================================================")
    print_t("Fitting model with cycle length {0}".format(cycle_lengths[k]))
    print_t("======================================================")
    model, history, (bgt, bgv) = fit_model(db_path+train_files[k], 
                               db_path+val_files[k],
                               db_path+test_files[k], 
                               batch_size=20, 
                               early_stop=True, 
                               early_stop_min_delta=1e-7, 
                               cycle_length=cycle_lengths[k],
                               n_epochs=20,
                               n_workers=96,
                               baseline=baseline)

    datafile_prefix = 'baseline' if baseline==True else 'simpledec'
    # Save history to a file
    dfhist = pd.DataFrame(history.history)
    dfhist.to_csv(datafile_prefix + train_files[k] + "_history_" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv")

    
    # Calculate roc_auc and save fpr, tpr as a DataFrame
    Xs = [bgv.__getitem__(k)[0] for k in range(min(len(bgv), int(np.ceil(4000/bgv.batch_size))))]
    ys = [bgv.__getitem__(k)[1] for k in range(min(len(bgv), int(np.ceil(4000/bgv.batch_size))))]
    
    Xs = np.vstack(Xs)
    ys = np.vstack(ys)
    
    y_pred = model.predict(Xs)
    fpr, tpr, thr = roc_curve(ys[:,0], y_pred[:,0])
    
    dfroc = pd.DataFrame({'fpr':np.array(fpr), 'tpr': np.array(tpr)})
    dfroc.to_csv(datafile_prefix + train_files[k] + "_roc_" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv")
    
    print_t("AUC score={0}".format(auc(fpr, tpr)))
