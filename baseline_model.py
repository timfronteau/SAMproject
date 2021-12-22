from keras.layers import Dense, Dropout, BatchNormalization
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
        self.nb_feat =   2048
        self.nb_of_label =   nb_of_label
        self.model = Sequential()
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self):
        m = Sequential()        
        m.add(Input(shape=self.nb_feat))
        m.add(Dense(self.nb_feat, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(1024, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(256, activation='relu'))
        m.add(Dropout(0.2))
        m.add(Dense(64, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(self.nb_of_label, activation = 'softmax'))

        m.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        self.model = m

    def train(self):
        self.model.fit( x=self.X_train, y=self.y_train, validation_data=(self.X_val, self.y_val),
                        batch_size=self.batch_size,
                        epochs=self.epochs)

    def load_model(self,path):
        self.model.load_model(path)
    
    def save(self,path):
        self.model.save(path)

    def evaluate(self):
        self.model.evaluate(x=self.X_test, y=self.y_test,
                            batch_size=self.batch_size)


