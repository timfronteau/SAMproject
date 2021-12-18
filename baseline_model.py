from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras import Input
import numpy as np

class Baseline():
    def __init__(self,X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size = 16, epochs = 10):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.freq_length =   np.shape(self.X_train)[1]
        self.time_length =   np.shape(self.X_train)[2]
        self.nb_of_label =   nb_of_label
        self.model = Sequential()
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self):
        model = Sequential()
        model.add(Input(shape = (self.freq_length,self.time_length)))
        model.add(Flatten())
        model.add(BatchNormalization)
        model.add(Dense(self.freq_length * self.time_length))
        model.add(Dropout(0.05))
        model.add(Dense(128))
        model.add(Dropout(0.05))
        model.add(Dense(self.nb_of_label, activation = 'softmax'))

        model.compile(loss = 'categorical_crossentropy', metrics = 'accuracy')
        
        self.model = model

    def train(self):
        self.model.fit( x=self.X_train, y=self.y_train, validation_data=(self.X_val, self.y_val),
                        batch_size=self.batch_size,
                        epochs=self.epochs)

