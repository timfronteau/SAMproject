from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras import Input


from baseline import Baseline


class BaselineImage(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)


    def build_model(self):
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