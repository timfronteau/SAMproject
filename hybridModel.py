
from keras.layers import Dense, Dropout, Concatenate
from keras.models import Model
from multiModel import MultiModel


class HybridModel(MultiModel):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size, epochs, input_shape)
        self.model_name = 'model_hybrid'

    def build_model(self): 
        clfs = self._build_sub_model(trainable=True) 
        embeddings1 = [Dense(self.nb_of_label, activation='relu', name=f'hybrid_embeddings_1_{idx}')(clf.layers[-1].output) for idx,clf in enumerate(clfs)]
        embeddings2 = [Dense(self.nb_of_label, activation='relu', name=f'hybrid_embeddings_2_{idx}')(emb) for idx,emb in enumerate(embeddings1)]
        merged = Concatenate(axis=1, name='hybrid_concat')(embeddings2)
        
        dense1 = Dense(self.nb_clf * self.nb_of_label, activation='relu', name='hybrid_dense_1')(merged)
        dropout1 = Dropout(0.05, name='hybrid_dropout_1')(dense1)
        dense2 = Dense(self.nb_clf * self.nb_of_label, activation='relu', name='hybrid_dense_2')(dropout1)
        dropout2 = Dropout(0.05, name='hybrid_dropout_2')(dense2)
        dense3 = Dense(32, activation = 'relu', name='hybrid_dense_3')(dropout2)
        dropout3 = Dropout(0.05, name='hybrid_dropout_3')(dense3)
        output = Dense(self.nb_of_label, activation='softmax', name='hybrid_output')(dropout3)

        inputs = [m.layers[0].input for m in clfs]
        hybrid_model = Model(inputs=inputs, outputs=output, name=self.model_name)

        hybrid_model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy']) # , optimizer=opt
        
        self.model = hybrid_model