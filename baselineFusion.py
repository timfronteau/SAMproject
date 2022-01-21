

from baseline import Baseline
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Sequential, Functional, Model
from tensorflow.keras.utils import plot_model
import numpy as np
from baseline import Baseline
from baselineAudio import BaselineAudio
from baselineImage import BaselineImage
from baselineMFCC import BaselineMFCC
from baselineText import BaselineText

class BaselineFusion(Baseline):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size, epochs, input_shape)
        
        self.DATA_TYPE_LIST = ['Image', 'Audio', 'MFCC']        
        self.nb_clf = len(self.DATA_TYPE_LIST) # number of classifiers`
        self.model_name = 'fusion_model'

    def build_model(self):
        clfs = []
        for DATA_TYPE in self.DATA_TYPE_LIST :
            if DATA_TYPE=='Image':
                DATASET_PATH = 'baseline_img.npz'
                MODEL_CLASS = BaselineImage
                MODEL_DIR = 'img_model'
                INPUT_SHAPE = (200, 200, 3)
            elif DATA_TYPE=='Audio':
                DATASET_PATH = 'baseline_deep.npz'
                MODEL_CLASS = BaselineAudio
                MODEL_DIR = 'deep_model'
                INPUT_SHAPE = 2048
            elif DATA_TYPE=='MFCC':
                DATASET_PATH = 'baseline_mfcc.npz'
                MODEL_CLASS = BaselineMFCC
                MODEL_DIR = 'mfcc_model'
                INPUT_SHAPE = 12*3
            elif DATA_TYPE=='text':
                DATASET_PATH = 'baseline_txt.npz'
                MODEL_CLASS = BaselineText
                MODEL_DIR = 'txt_model'
                INPUT_SHAPE = 5000

            dataset_baseline = np.load(DATASET_PATH)
            X_train = dataset_baseline['X_train']
            X_val = dataset_baseline['X_val']
            X_test = dataset_baseline['X_test']
            y_train = dataset_baseline['y_train']
            y_val = dataset_baseline['y_val']
            y_test = dataset_baseline['y_test']

            BATCH_SIZE = 32
            EPOCHS = 10

            baseline = MODEL_CLASS(X_train, X_val, X_test, y_train, y_val, y_test,
                            self.nb_of_label, batch_size=BATCH_SIZE, epochs=EPOCHS, input_shape=INPUT_SHAPE)
            baseline.load_model(MODEL_DIR)
            m = Model(baseline.model.inputs, baseline.model.output, name=f'{MODEL_DIR}')
            m.trainable = False
            clfs.append(m)
            print(m.summary())

        #inputs = [Input(shape=(1, )) for _ in self.nb_clf]    
        embeddings1 = [Dense(self.nb_of_label, activation='relu', name=f'fusion_embeddings_1_{idx}')(clf.layers[-1].output) for idx,clf in enumerate(clfs)]
        embeddings2 = [Dense(self.nb_of_label, activation='relu', name=f'fusion_embeddings_2_{idx}')(emb) for idx,emb in enumerate(embeddings1)]
        merged = Concatenate(axis=1, name='fusion_concat')(embeddings2)
        
        dense1 = Dense(self.nb_clf * self.nb_of_label, activation='relu', name='fusion_dense_1')(merged)
        dropout1 = Dropout(0.05, name='fusion_dropout_1')(dense1)
        dense2 = Dense(self.nb_clf * self.nb_of_label, activation='relu', name='fusion_dense_2')(dropout1)
        dropout2 = Dropout(0.05, name='fusion_dropout_2')(dense2)
        dense3 = Dense(32, activation = 'relu', name='fusion_dense_3')(dropout2)
        dropout3 = Dropout(0.05, name='fusion_dropout_3')(dense3)
        output = Dense(self.nb_of_label, activation='softmax', name='fusion_output')(dropout3)

        inputs = [m.layers[0].input for m in clfs]
        m = Model(inputs=inputs, outputs=output, name=self.model_name)
        print(m.summary())
        m.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy']) # , optimizer=opt
        plot_model(model=m, to_file=f'{self.model_name}_tree.png', show_shapes=True)
        self.model = m



      