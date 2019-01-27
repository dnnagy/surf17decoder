from QEC_full import print_t
from QEC_full import fit_model
from sklearn.metrics import roc_curve, auc

cycle_lengths = [100, 150, 200]

train_files = ['big_c{0}_train.db'.format(k) for k in cycle_lengths]
val_files = ['big_c{0}_validation.db'.format(k) for k in cycle_lengths]
test_files = ['big_c{0}_test.db'.format(k) for k in cycle_lengths]

db_path = './data/'

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
                               n_workers=96)
    
    """
        Plot roc auc for the best performing model
    """
    X, y = bgt.__getitem__(0)
    y_pred = model.predict(X)
    fpr, tpr, thr = roc_curve(y, y_pred)
    print_t("AUC score={0}".format(auc(fpr, tpr)))
    raise StopIteration("End here.")
    """
        Plot val_acc vs epoch number based on history
    """
