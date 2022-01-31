from keras.models import Sequential
import abc
import pandas as pd
import time
import csv
import os 
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report as clf_report
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

class Model():
    def __init__(self,X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.input_shape = input_shape
        self.nb_of_label = nb_of_label
        self.model = Sequential()
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        self.model_name = ""

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError
    
    def fit(self):
        self.history = self.model.fit(x=self.X_train, y=self.y_train,
                                validation_data=(self.X_val, self.y_val),
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                use_multiprocessing=True)
    def predict(self, x):
        return self.model.predict(x, batch_size=self.batch_size)

    def load_model(self,path):
        self.model = load_model(f"saved_model/{path}")
    
    def save(self,path):
        try : self.model.save(f"saved_model/{path}")
        except : 
            if not os.path.exists('saved_model'):
                os.makedirs('saved_model')

    def evaluate(self):
        print('Model evaluation:')
        self.model.evaluate(x=self.X_test, y=self.y_test,
                            batch_size=self.batch_size)

    def model_result(self):
        print("\nRandom model result ...")
        self.evaluate()

    def save_history_to_csv(self, MODEL_DIR, ITERATION):
        try:os.makedirs(MODEL_DIR)
        except FileExistsError: pass
        with open(f'{MODEL_DIR}/history.csv', mode='a') as f:
            hist = pd.DataFrame(self.history.history).reset_index()
            hist['index'] = hist['index'].apply(lambda x: ITERATION*self.epochs + x)
            hist.to_csv(f, index=False, header=False)

    def save_config_to_csv(self, MODEL_DIR, BATCH_SIZE, EPOCHS,NB_ITERATIONS, DATA_TYPE, DATASET_PATH, MODEL_CLASS, INPUT_SHAPE):
        try:os.makedirs(MODEL_DIR)
        except FileExistsError: pass
        with open(f'{MODEL_DIR}/history.csv', mode='w') as f:
            writer = csv.writer(f)
            writer.writerows([   
                            [f"DATE = {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime())}"], 
                            [f"BATCH_SIZE = {BATCH_SIZE}"], 
                            [f"EPOCHS = {EPOCHS}"], 
                            [f"NB_ITERATIONS = {NB_ITERATIONS}"], 
                            [f"DATA_TYPE = {DATA_TYPE}"], 
                            [f"DATASET_PATH = {DATASET_PATH}"], 
                            [f"MODEL_CLASS = {MODEL_CLASS}"], 
                            [f"MODEL_DIR = {MODEL_DIR}"], 
                            [f"INPUT_SHAPE = {INPUT_SHAPE}"],
                            ['epochs','loss','accuracy','val_loss','val_accuracy']
                            ])
    def summary(self):
        return self.model.summary()

    def plot_tree_model(self):
        print(f'Saving tree model ...', end='')
        plot_model(model=self.model, to_file=f'figures/tree_model/{self.model_name}_tree.png', dpi=300, show_shapes=True)
        print('done')
    

    def confusion_mat(self, X_test, y_test):     
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred , axis=1) 
        y_test = np.argmax(y_test, axis=1)
        return confusion_matrix(y_test, y_pred)
    
    def plot_confusion_matrix(self, X_test, y_test, classes=None):
        print(f'Saving confusion matrix ...', end='')
        cm = self.confusion_mat(X_test, y_test)
        plt.figure(figsize=(30,20))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=classes)
        disp.plot()
        plt.title(f'Confusion Matrix for {self.model_name}')
        plt.savefig(f"figures/confusion_matrix/conf_mat_{self.model_name}.png", dpi=300)
        print('done')

    def classification_report(self, X_test, y_test, save=False):   
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred , axis=1) 
        y_test = np.argmax(y_test, axis=1)
        res = clf_report(y_test, y_pred)
        if save : 
            dico = clf_report(y_test, y_pred, output_dict=save)
            with open(f"classification_report/{self.model_name}.pkl", 'wb') as f:
                pkl.dump(dico, f)
        print(f"Classifation report for {self.model_name};\n{res}")
        return res
        
    def load_classification_report(self):
        with open(f"classification_report/{self.model_name}.pkl", 'rb') as f:
            dico = pkl.load(f)
        return dico


