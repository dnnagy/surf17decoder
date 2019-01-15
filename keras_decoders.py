import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Flatten, Concatenate, Dropout
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import time

def print_t(str_):
  ## 24 hour format ##
  return print( "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + str_)

""" 
This is a simple decoder, that gets only the syndromes (without final syndromes)
and the measured parity. 
"""
class SimpleDecoder:
    def __init__(self, xshape, hidden_size=64):
        self.hidden_size=hidden_size
        self.xshape=xshape
        pass
    
    def create_model(self):
        # This returns a tensor
        input_syndr = Input(shape=(self.xshape))
        
        x = LSTM(self.hidden_size, return_sequences=True)(input_syndr)
        x = LSTM(self.hidden_size, return_sequences=True)(x)
        #x = Dropout(0.5)(x)
        x = LSTM(self.hidden_size, return_sequences=True)(x)
        x = LSTM(self.hidden_size, return_sequences=True)(x)
        #x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        #x = Dropout(0.25)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # optimizer
        # lr plot 
        sgd = SGD(lr = 0.01, momentum = 0.9, decay = 0, nesterov = False)
        
        model = Model(inputs=input_syndr, outputs=predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


"""
This is a more advanced model with 2 branches and 
Merge layer.
"""
class BranchedDecoder:
    pass