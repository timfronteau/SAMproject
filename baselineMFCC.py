from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras import Input
from tensorflow.keras.optimizers import Adam

from baseline import Baseline

class BaselineMFCC(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)

    def build_model(self):
        m = Sequential()        
        m.add(Input(shape=self.input_shape))
        m.add(Dense(16, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(16, activation='relu'))
        m.add(Dropout(0.05))
        m.add(Dense(self.nb_of_label, activation='softmax'))

        m.summary()
        # opt = Adam(learning_rate=0.01)
        m.compile(loss='categorical_crossentropy', metrics=['accuracy']) # , optimizer=opt
        
        self.model = m