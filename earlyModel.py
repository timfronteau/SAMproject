from keras.layers import Dense, Dropout, Average, Concatenate, Input
from keras.models import Model

from baseline import Baseline
from baselineAudio import BaselineAudio
from baselineImage import BaselineImage
from baselineMFCC import BaselineMFCC
from baselineText import BaselineText

class EarlyModel(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        self.imageModel = BaselineImage(X_train, X_val, X_test, y_train, y_val, y_test,
                        nb_of_label, batch_size, epochs, (200, 200, 3))

    def build_model(self):

        self.imageModel.load_model("img_model")
        m1 = Model(self.imageModel.model.inputs, self.imageModel.model.output, name=f"img_model")
        i2 = Input(shape=2048)
        i3 = Input(shape=12*3)
        i4 = Input(shape=5000)
        inputs = [m1.layers[0].input, i2, i3, i4]
        outputs = [m1.layers[8].output, i2, i3, i4]
        print(m1.layers[8].output)
        print(outputs)
        input_mod = Model(inputs=inputs, outputs=outputs)
        cct = Concatenate()(input_mod.outputs)
        d1 = Dense(64, activation='relu')(cct)
        drop1 = Dropout(0.1)(d1)
        d2 = Dense(64, activation='relu')(drop1)
        drop2 = Dropout(0.1)(d2)
        d3 = Dense(32, activation='relu')(drop2)
        drop3 = Dropout(0.1)(d3)
        d4 = Dense(self.nb_of_label, activation='softmax')(drop3)
        late_model = Model(inputs=inputs, outputs=d4)  # avg d4 cct

        late_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        print(late_model.summary())

        self.model = late_model