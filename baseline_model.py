from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras import Input
from tensorflow.keras.optimizers import Adam
import numpy as np

class Baseline():
    def __init__(self,X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        # self.nb_feat = 2048
        self.input_shape = input_shape
        self.nb_of_label = nb_of_label
        self.model = Sequential()
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self):
        m = Sequential()        
        m.add(Input(shape=self.input_shape))
        if type(self.input_shape) == tuple:
            m.add(MaxPooling2D((4, 4)))
            m.add(Flatten())
        m.add(Dense(2048, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(1024, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(256, activation='relu'))
        m.add(Dropout(0.2))
        m.add(Dense(64, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(self.nb_of_label, activation='softmax'))

        # opt = Adam(learning_rate=0.01)
        m.compile(loss='categorical_crossentropy', metrics=['accuracy']) # , optimizer=opt
        
        self.model = m

    def build_conv_model(self):
        m = Sequential()
        m.add(Input(shape=self.input_shape))
        m.add(MaxPooling2D((4, 4)))
        m.add(Conv2D(16, (3, 3), activation='relu'))
        m.add(MaxPooling2D((2, 2)))
        m.add(Conv2D(32, (3, 3), activation='relu'))
        m.add(MaxPooling2D((2, 2)))
        m.add(Conv2D(32, (3, 3), activation='relu'))
        size = int((int((int(self.input_shape[0]/4)-2)/2)-2)/2)
        m.add(AveragePooling2D((size, size), padding='same'))
        m.add(Flatten())
        m.add(BatchNormalization())
        m.add(Dropout(0.1))
        m.add(Dense(100, activation='relu'))
        m.add(Dropout(0.1))
        m.add(Dense(100, activation='relu'))
        m.add(Dropout(0.1))
        m.add(Dense(self.nb_of_label, activation='relu'))
        m.add(Dense(self.nb_of_label, activation='softmax'))


        m.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        m.summary()
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


