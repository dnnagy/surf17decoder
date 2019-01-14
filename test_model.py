import keras

# Generate all the data
from QECDataGenerator import QECDataGenerator

datagen=QECDataGenerator(filename_base='small', train_size=2*10**4, 
                         validation_size=4*10**3, test_size=10**2, verbose=1)

# generate train data
#training_fname=datagen.generate(0) 
training_fname='./data/small_train.db'

#generate validation data
#validation_fname=datagen.generate(1)
validation_fname='./data/small_validation.db'

#generate testing data
#test_fname=datagen.generate(2)
test_fname='./data/small_test.db'

# Initialize batch generator
from SQLBatchGenerators import SimpleBatchGenerator

bg=SimpleBatchGenerator(training_fname, validation_fname, test_fname, batch_size=5000, mode='training')
bgv=SimpleBatchGenerator(training_fname, validation_fname, test_fname, batch_size=2000, mode='validation')

# Test models
from keras_decoders import SimpleDecoder

kd=SimpleDecoder(xshape=(20,8), hidden_size=64)
model=kd.create_model()
model.summary()

# Implement callback
from sklearn.metrics import roc_curve, auc
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import time

def print_t(str_):
  ## 24 hour format ##
  return print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str_)

class test_callback(Callback):
  def __init__(self):
    self.X = bgv.__getitem__(0)[0]
    self.y = bgv.__getitem__(0)[1]
  
  def on_train_begin(self, logs={}):
    return

  def on_train_end(self, logs={}):
    return
  
  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    print_t("Generating roc curve for epoch #{0} ...".format(epoch))
    
    y_pred = self.model.predict(self.X)
    print_t("X.shape={0}".format(self.X.shape))
    print_t("y_pred.shape={0}".format(y_pred.shape))
    fpr, tpr, thr = roc_curve(self.y, y_pred)
    
    auc_score = auc(fpr, tpr)
    
    plt.ioff() ## Turn off interactive mode
    plt.figure(figsize=(10,6), dpi=196)
    plt.plot(fpr, tpr, label='SimpleDecoder, auc={0}'.format(auc_score))
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend()
    plt.savefig("SimpleDecoder_e{0}_roc.png".format(epoch))
    print_t("Epoch {0} roc-auc: {1}".format(epoch, str(round(auc_score,4))))
    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return

### FIT MODEL
hist=model.fit_generator(generator=bg,
                    epochs=2,
                    validation_data=bgv,
                    use_multiprocessing=True,
                    callbacks=[test_callback()],
                    workers=2);
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['val_acc'], label='val_acc')
plt.savefig("SimpleDecoder_accs.png")
print_t("DONE.")