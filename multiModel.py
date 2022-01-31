
from keras import models
from model import Model
from utils import extract_config

class MultiModel(Model):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size, epochs, input_shape)
        
        self.DATA_TYPE_LIST = ['Image', 'Audio', 'MFCC', 'Text']        
        self.nb_clf = len(self.DATA_TYPE_LIST) # number of classifiers`

    def _build_sub_model(self, trainable:bool=False):
        """"
        Cette methode permet de creer de récupérer les classifieurs unimodaux"""
        clfs = []
        for DATA_TYPE in self.DATA_TYPE_LIST :
            _, MODEL_CLASS, MODEL_DIR,  INPUT_SHAPE = extract_config(DATA_TYPE, MULTI_TYPE='Uni')            

            model = MODEL_CLASS([],[],[],[],[],[],
                            self.nb_of_label, input_shape=INPUT_SHAPE)
            try:model.load_model(MODEL_DIR)
            except: 
                print(f'Aucun model trouvé, un nouveau model sera sauvegardé sous le nom de : {MODEL_DIR}')
                model.fit()
            m = models.Model(model.model.inputs, model.model.output, name=f'{MODEL_DIR}')
            m.trainable = trainable
            clfs.append(m)
        return clfs

      