from keras.layers import Dense, Dropout, BatchNormalization, GRU
from keras.models import Sequential
from keras import Input


from baseline import Baseline


class BaselineText(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)

    def build_model(self):
        m = Sequential(name='model_text')
        m.add(Input(shape=self.input_shape, name='text_input'))
        m.add(Dropout(0.3, name='text_dropout_1'))
        m.add(Dense(1024, activation='relu', name='text_dense_1'))
        m.add(Dropout(0.3, name='text_dropout_2'))
        m.add(Dense(256, activation='relu', name='text_dense_2'))
        m.add(Dropout(0.1, name='text_dropout_3'))
        m.add(Dense(self.nb_of_label, activation='relu', name='text_dense_3'))
        m.add(Dense(self.nb_of_label, activation='softmax', name='text_output'))

        m.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        m.summary()
        self.model = m