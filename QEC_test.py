from QEC_full import print_t
from QEC_full import fit_model

cycle_lengths = [100, 150, 200]

train_files = ['small_c{0}_train.db'.format(k) for k in cycle_lengths]
val_files = ['small_c{0}_validation.db'.format(k) for k in cycle_lengths]
test_files = ['small_c{0}_test.db'.format(k) for k in cycle_lengths]

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
    print_t("Batch generated data with shapes: X.shape={0}, y.shape={0}".format(X.shape))
    raise StopIteration("End here.")
    """
        Plot val_acc vs epoch number based on history
    """
