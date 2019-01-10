import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Concatenate
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

# This is a very simple model for decoding, containing only
# 2 LSTM layers and a Dense layer.
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
        x = Dense(self.hidden_size, activation='relu')(x)
        predictions = Dense(1, activation='softmax')(x)
        
        # optimizer
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        
        model = Model(inputs=input_syndr, outputs=predictions)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

class BranchedDecoder:
    pass