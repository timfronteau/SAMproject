from keras.layers import Dense, Dropout, BatchNormalization, GRU
from keras.models import Sequential
from keras import Input


from baseline import Baseline


class BaselineText(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)


    def build_model(self):
        m = Sequential()
        m.add(Input(shape=self.input_shape))
        m.add(Dropout(0.3))
        m.add(Dense(1024, activation='relu'))
        m.add(Dropout(0.3))
        m.add(Dense(256, activation='relu'))
        m.add(Dropout(0.1))
        m.add(Dense(self.nb_of_label, activation='relu'))
        m.add(Dense(self.nb_of_label, activation='softmax'))

        m.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        m.summary()
        self.model = m