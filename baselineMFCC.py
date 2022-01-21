from unicodedata import name
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras import Input
from tensorflow.keras.optimizers import Adam

from baseline import Baseline

class BaselineMFCC(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)

    def build_model(self):
        m = Sequential(name='model_mfcc')        
        m.add(Input(shape=self.input_shape, name='mfcc_input'))
        m.add(Dense(32, activation='relu', name='mff_dense_1'))
        m.add(Dropout(0.05, name='mfcc_dropout_1'))
        m.add(Dense(16, activation='relu', name='mfcc_dense_2'))
        m.add(Dropout(0.05, name='mfcc_dropout_2'))
        m.add(Dense(self.nb_of_label, activation='softmax', name='mfcc_output'))

        m.summary()
        # opt = Adam(learning_rate=0.01)
        m.compile(loss='categorical_crossentropy', metrics=['accuracy']) # , optimizer=opt
        
        self.model = m