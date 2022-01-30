from keras.layers import Dense, Dropout, Concatenate
from keras.models import Model
from multiModel import MultiModel


class LateModel(MultiModel):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        self.model_name = 'model_late'

    def build_model(self):
        clfs = self._build_sub_model() 

        inputs = [m.layers[0].input for m in clfs]
        outputs = [m.layers[-1].output for m in clfs]
        input_mod = Model(inputs=inputs, outputs=outputs)
        cct = Concatenate(axis=1, name='late_concat')(input_mod.outputs)  
        dense1 = Dense(64, activation = 'relu', name='late_dense_1')(cct)
        dropout1 = Dropout(0.3, name='late_dropout_1')(dense1)
        dense2 = Dense(32, activation = 'relu', name='late_dense_2')(dropout1)
        dropout2 = Dropout(0.3, name='late_dropout_2')(dense2)
        output = Dense(self.nb_of_label, activation='softmax', name='late_output')(dropout2)
        late_model = Model(inputs=inputs, outputs=output, name=self.model_name)

        late_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.model = late_model