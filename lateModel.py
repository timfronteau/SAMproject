from keras.layers import Average
from keras.models import Model
from multiModel import MultiModel

class LateModel(MultiModel):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        self.model_name = 'model_late'

    def build_model(self):
        clfs = self._build_sub_model(trainable=True) 
        inputs = [m.layers[0].input for m in clfs]
        outputs = [m.layers[-1].output for m in clfs]
        input_mod = Model(inputs=inputs, outputs=outputs)
        output = Average(name='late_avg')(input_mod.outputs)
        late_model = Model(inputs=inputs, outputs=output, name=self.model_name)

        late_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.model = late_model
