from keras.layers import Dense, Dropout, Concatenate, Input
from keras.models import Model

from multiModel import MultiModel

class EarlyModel(MultiModel):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=5000):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        self.model_name = 'model_early'

    def build_model(self):
        clfs = self._build_sub_model(trainable=True)
        img_idx = 0
        inputs = [clf.layers[0].input for clf in clfs]        
        inputs[img_idx] = clfs[img_idx].layers[0].input                    
        embedding1 = [clf.layers[0].output for clf in clfs]    
        embedding1[img_idx] = clfs[img_idx].layers[8].output   
        
        embedding2 = Model(inputs=inputs, outputs=embedding1, name='early_embedding_2')
       
        cct = Concatenate(axis=1, name='early_concat')(embedding2.outputs)        
        d1 = Dense(2048, activation='relu', name='early_dense_1')(cct)
        drop1 = Dropout(0.3, name='early_dropout_1')(d1)
        d2 = Dense(256, activation='relu', name='early_dense_2')(drop1)
        drop2 = Dropout(0.3, name='early_dropout_2')(d2)
        d3 = Dense(32, activation='relu', name='early_dense_3')(drop2)
        drop3 = Dropout(0.3, name='early_dropout_3')(d3)
        output = Dense(self.nb_of_label, activation='softmax', name='early_dense_4')(drop3)
        early_model = Model(inputs=inputs, outputs=output)

        early_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model = early_model