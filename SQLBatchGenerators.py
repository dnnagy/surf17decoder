import copy
import sqlite3
import numpy as np
import scipy.optimize as optim

import keras
"""
keras.utils.Sequence is the base object for fitting to a sequence of data, such as a dataset.
Every Sequence must implement the __getitem__ and the __len__ methods. 
If you want to modify your dataset between epochs you may implement on_epoch_end. 
The method __getitem__ should return a complete batch.
"""

""" 
The following is a simple batch generator, that generates only the syndromes (without final syndromes)
and the measured parity 
"""
class SimpleBatchGenerator(keras.utils.Sequence):
  def __init__(self, training_fname, validation_fname, test_fname, batch_size=16, mode='training'):
    
    self.dim_syndr = 8
    self.n_steps_net2 = 4
    
    self.training_fname=training_fname
    self.validation_fname=validation_fname
    self.test_fname=test_fname
    
    self.batch_size=batch_size

    if mode not in ['training', 'validation', 'test']:
      raise ValueError("mode must be either 'training', 'validation' or 'test'")
      
    self.mode=mode
    return
  
  def _load_data(self):
    # Establish connections
    self.training_conn = sqlite3.connect(self.training_fname)
    self.validation_conn = sqlite3.connect(self.validation_fname)
    self.test_conn = sqlite3.connect(self.test_fname)
    
    training_c = self.training_conn.cursor()
    validation_c = self.validation_conn.cursor()
    test_c = self.test_conn.cursor()
    
    # get all the seeds
    training_c.execute('SELECT seed FROM data')
    validation_c.execute('SELECT seed FROM data')
    test_c.execute('SELECT seed FROM data')
        
    self.training_keys = list(sorted([s[0] for s in training_c.fetchall()]))
    self.validation_keys = list(sorted([s[0] for s in validation_c.fetchall()]))
    self.test_keys = list(sorted([s[0] for s in test_c.fetchall()]))

    # checks that there is no overlapp in the seeds of the data sets
    self.N_training = len(self.training_keys)
    self.N_validation = len(self.validation_keys)
    self.N_test = len(self.test_keys)
    all_keys = set(self.training_keys + self.validation_keys + self.test_keys)
        
    if len(all_keys) < self.N_training + self.N_validation + self.N_test:
      raise ValueError("There is overlap between the seeds of the training,  validation, and test sets. This"
                         "is bad practice")
      print("loaded databases and checked exclusiveness training, "
              "validation, and test keys")
      print("self.N_training=" + str(self.N_training) + ", self.N_validaiton=" + 
            str(self.N_validation) + ", self.N_test=" + str(self.N_test) + ".")
    return

  def _close_databases(self):
    """ This function closes all databases """
    self.training_conn.close()
    self.validation_conn.close()
    self.test_conn.close()
    return

  def _fetch_one_record(self):
    # select data from the corresponding database
    if self.mode == "training":
      c = self.training_conn.cursor()
    elif self.mode == "validation":
      c = self.validation_conn.cursor()
    elif self.mode == "test":
      c = self.test_conn.cursor()
    else:
      raise ValueError("The only allowed data_types are: 'training','validation' and 'test'.")
    
    query="SELECT events, err_signal, parity, length FROM data ORDER BY RANDOM() LIMIT 1"
    c.execute(query)
    # c.execute("SELECT events, err_signal, parity, length FROM data ORDER BY RANDOM() LIMIT ?", (1, ))
    sample = c.fetchmany(1)
    
    return sample
  
  # fetch n records where the final parity is not null
  def _fetch_records_nonull(self, n):
    
    try:
      self.tarining_conn
      self.validation_conn
      self.test_conn
    except:
      self._load_data()
    
    if self.mode == "training":
      c = self.training_conn.cursor()
    elif self.mode == "validation":
      c = self.validation_conn.cursor()
    elif self.mode == "test":
      c = self.test_conn.cursor()
    else:
      raise ValueError("The only allowed data_types are: 'training','validation' and 'test'.")
    
    query="SELECT events, err_signal, parity, length FROM data WHERE hex(parity)='01' LIMIT " + str(n)
    c.execute(query)
    samples=c.fetchmany(n)
    return samples

  def _fetch_one_batch(self):
    # select data from the corresponding database
    if self.mode == "training":
      c = self.training_conn.cursor()
    elif self.mode == "validation":
      c = self.validation_conn.cursor()
    elif self.mode == "test":
      c = self.test_conn.cursor()
    else:
      raise ValueError("The only allowed data_types are: 'training','validation' and 'test'.")
    
    query="SELECT events, err_signal, parity, length FROM data ORDER BY RANDOM() LIMIT " + str(self.batch_size)
    c.execute(query)
    # c.execute("SELECT events, err_signal, parity, length FROM data ORDER BY RANDOM() LIMIT ?", (self.batch_size, ))
    samples = c.fetchmany(self.batch_size)
    
    return samples
  
  def _convert_sample(self, sample):
    """ formats a single batch of data
    
    Input
    -----
    
    sample - raw data from the database
    """
    
    syndr, fsyndr, parity, length = sample
    n_steps = int(len(syndr) / self.dim_syndr)
    
    # # format into shape [steps, syndromes]
    # syndr1 = np.fromstring(syndr, dtype=bool).reshape([n_steps, -1])
    
    # # get and set length information
    # len1 = np.frombuffer(length, dtype=int)[0]
    
    # # the second length is set by n_steps_net2, except if len1 is shorter
    # len2 = min(len1, self.n_steps_net2)
    
    # syndr2 = syndr1[len1 - len2:len1 - len2 + self.n_steps_net2]
    # fsyndr = np.fromstring(fsyndr, dtype=bool)
    # parity = np.frombuffer(parity, dtype=bool)
    
    syndr = np.fromstring(syndr, dtype=bool).reshape([n_steps, -1])
    parity = np.frombuffer(parity, dtype=bool)
    
    return syndr, parity

  def get_n_batches(self, n_batches):
    self._load_data()
    batches=[]
    for k in range(n_batches):
      fetched_samples=self._fetch_one_batch()
      batch=[]
      for j in range(len(fetched_samples)):
        record=self._convert_sample(fetched_samples[j])
        batch.append(record)
      batches.append(batch)
    return batches

  def get_data_shape(self):
    # self._load_data()
    # samples=self._fetch_one_batch(10)
    # batch=self._get_batch_from_sample(samples[0])
    # self._close_databases()
    # return batch
    pass 

  """
  Functions to be implemented according to keras.utils.Sequence.
  """
  # A keras.utils.Sequence object must impement __len__ function
  def __len__(self):
    try:
      self.N_training
      self.N_validation
      self.N_test
    except:
      self._load_data()
    
    if self.mode == "training":
      return int(np.ceil(self.N_training/float(self.batch_size)))
    elif self.mode == "validation":
      return int(np.ceil(self.N_validation/float(self.batch_size)))
    elif self.mode == "test":
      return int(np.ceil(self.N_test/float(self.batch_size)))
    return
  
  # A keras.utils.Sequence object must impement __getitem__ function
  def __getitem__(self, index):
    
    try:
      self.tarining_conn
      self.validation_conn
      self.test_conn
    except:
      self._load_data()
        
    if self.mode == "training":
      c = self.training_conn.cursor()
    elif self.mode == "validation":
      c = self.validation_conn.cursor()
    elif self.mode == "test":
      c = self.test_conn.cursor()
    else:
      raise ValueError("The only allowed data_types are: 'training','validation' and 'test'.")
    
    query="SELECT events, err_signal, parity, length FROM data ORDER BY RANDOM() LIMIT " + str(self.batch_size) + " OFFSET " + str(index*self.batch_size)
    
    c.execute(query)
    samples = c.fetchall()

    #This is the indexth batch
    X_batch=[]
    y_batch=[]
    for j in range(self.batch_size):
      X, y = self._convert_sample(samples[j])
      X_batch.append(X)
      y_batch.append(y)
    return (np.array(X_batch), np.array(y_batch))



"""
        END 
"""

"""
    This is a more sophisticated batch generator, that reads both syndr1 and final syndrome.
    This generator is unfinished
"""
# class FullBatchGenerator(keras.utils.Sequence):
#   def __init__(self, training_fname, validation_fname, test_fname, mode):
    
#     self.dim_syndr = 8
#     self.n_steps_net2 = 4
    
#     self.training_fname=training_fname
#     self.validation_fname=validation_fname
#     self.test_fname=test_fname
    
#     if mode not in ['training', 'validation', 'test']:
#       raise ValueError("mode must be either 'training', 'validation' or 'test'")
      
#     self.mode=mode
#     return
  
#   def _load_data(self):
#     # Establish connections
#     self.training_conn = sqlite3.connect(self.training_fname)
#     self.validation_conn = sqlite3.connect(self.validation_fname)
#     self.test_conn = sqlite3.connect(self.test_fname)
    
#     training_c = self.training_conn.cursor()
#     validation_c = self.validation_conn.cursor()
#     test_c = self.test_conn.cursor()
    
#     # get all the seeds
#     training_c.execute('SELECT seed FROM data')
#     validation_c.execute('SELECT seed FROM data')
#     test_c.execute('SELECT seed FROM data')
        
#     self.training_keys = list(sorted([s[0] for s in training_c.fetchall()]))
#     self.validation_keys = list(sorted([s[0] for s in validation_c.fetchall()]))
#     self.test_keys = list(sorted([s[0] for s in test_c.fetchall()]))

#     # checks that there is no overlapp in the seeds of the data sets
#     self.N_training = len(self.training_keys)
#     self.N_validation = len(self.validation_keys)
#     self.N_test = len(self.test_keys)
#     all_keys = set(self.training_keys + self.validation_keys + self.test_keys)
        
#     if len(all_keys) < self.N_training + self.N_validation + self.N_test:
#       raise ValueError("There is overlap between the seeds of the training,  validation, and test sets. This"
#                          "is bad practice")
#       print("loaded databases and checked exclusiveness training, "
#               "validation, and test keys")
#       print("self.N_training=" + str(self.N_training) + ", self.N_validaiton=" + 
#             str(self.N_validation) + ", self.N_test=" + str(self.N_test) + ".")
#     return

#   def _close_databases(self):
#     """ This function closes all databases """
#     self.training_conn.close()
#     self.validation_conn.close()
#     self.test_conn.close()
#     return

#   def _fetch_one_batch(self, batch_size, oversample=False):
    
#     # select data from the corresponding database
#     if self.mode == "training":
#       c = self.training_conn.cursor()
#     elif self.mode == "validation":
#       c = self.validation_conn.cursor()
#     elif self.mode == "test":
#       c = self.test_conn.cursor()
#     else:
#       raise ValueError("The only allowed data_types are: 'training','validation' and 'test'.")
      
#     if oversample:
#       c.execute("SELECT events, err_signal, parities FROM data ORDER BY RANDOM() LIMIT ?",
#                 (batch_size, ))
#     else:
#       c.execute("SELECT events, err_signal, parity, length FROM data ORDER BY RANDOM() LIMIT ?",
#                 (batch_size, ))
      
#     samples = c.fetchmany(batch_size)
#     return samples
  
#   def _get_batch_from_sample(self, sample, oversample=False):
#     """ formats a single batch of data
    
#     Input
#     -----
    
#     sample - raw data from the database
#     """
#     syndr, fsyndr, parity, length = sample
#     n_steps = int(len(syndr) / self.dim_syndr)
    
#     # format into shape [steps, syndromes]
#     syndr1 = np.fromstring(syndr, dtype=bool).reshape([n_steps, -1])
    
#     # get and set length information
#     len1 = np.frombuffer(length, dtype=int)[0]
    
#     # the second length is set by n_steps_net2, except if len1 is shorter
#     len2 = min(len1, self.n_steps_net2)
    
#     syndr2 = syndr1[len1 - len2:len1 - len2 + self.n_steps_net2]
#     fsyndr = np.fromstring(fsyndr, dtype=bool)
#     parity = np.frombuffer(parity, dtype=bool)
    
#     return syndr1, syndr2, fsyndr, len1, len2, parity
  
    
#   def get_data_shape(self):
#     self._load_data()
#     samples=self._fetch_one_batch(10)
#     batch=self._get_batch_from_sample(samples[0])
#     self._close_databases()
#     return batch

#   """
#   Functions to be implemented according to keras.utils.Sequence.
#   """
#   # A keras.utils.Sequence object must impement __len__ function
#   def __len__(self):
#     if self.mode == "training":
#       return self.N_training
#     elif self.mode == "validation":
#       return self.N_validation
#     elif self.mode == "test":
#       return self.N_test
#     return
  
#   # A keras.utils.Sequence object must impement __getitem__ function
#   def __getitem__(self, index):
#     X = np.random.randint(0,10, size=(self.batch_size, *self.data_shape))
#     y = np.array([np.random.choice([0,1]) for _ in range(self.batch_size)])
#     return