from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Average
from keras.models import Sequential, Functional, Model
from keras import Input
from tensorflow.keras.optimizers import Adam

from baseline import Baseline
from baselineAudio import BaselineAudio
from baselineImage import BaselineImage
from baselineMFCC import BaselineMFCC
from baselineText import BaselineText


class LateModel(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        self.imageModel = BaselineImage(X_train, X_val, X_test, y_train, y_val, y_test,
                        nb_of_label, batch_size, epochs, (200, 200, 3))
        self.audioModel = BaselineAudio(X_train, X_val, X_test, y_train, y_val, y_test,
                        nb_of_label, batch_size, epochs, 2048)
        self.MFCCModel = BaselineMFCC(X_train, X_val, X_test, y_train, y_val, y_test,
                        nb_of_label, batch_size, epochs, 12*3)
        self.textModel = BaselineText(X_train, X_val, X_test, y_train, y_val, y_test,
                        nb_of_label, batch_size, epochs, 5000)
        self.model = Functional()

    def build_model(self):
        # big_input_shape = self.imageModel.input_shape + self.audioModel.input_shape + self.MFCCModel.input_shape + self.textModel.input_shape
        # m.add(Input(shape=big_input_shape))
        # img = Model(inputs=self.imageModel.model.input, outputs=self.imageModel.model.output)
        # aud = Model(inputs=self.audioModel.model.input, outputs=self.audioModel.model.output)
        # mfcc = Model(inputs=self.MFCCModel.model.input, outputs=self.MFCCModel.model.output)
        # txt = Model(inputs=self.textModel.model.input, outputs=self.textModel.model.output)
        models = [self.imageModel.model, self.audioModel.model, self.MFCCModel.model, self.textModel.model]
        inputs = [m.layers[0].input for m in models]
        outputs = [m.layers[-1].output for m in models]
        input_mod = Model(inputs=inputs, outputs=outputs)
        avg = Average()(input_mod)
        d1 = Dense(32, activation='relu')(avg)
        drop1 = Dropout(0.1)(d1)
        d2 = Dense(64, activation='relu')(drop1)
        drop2 = Dropout(0.1)(d2)
        d3 = Dense(32, activation='relu')(drop2)
        drop3 = Dropout(0.1)(d3)
        d4 = Dense(self.nb_of_label, activation='softmax')(drop3)
        late_model = Model(inputs=inputs, outputs=d4)

        late_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.model = late_model




