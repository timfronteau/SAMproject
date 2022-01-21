from keras.models import Sequential
import abc
import pandas as pd
import time
import csv
import os 
from keras.models import load_model

class EarlyModel():
    def __init__(self,X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=2048):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        # self.nb_feat = 2048
        self.input_shape = input_shape
        self.nb_of_label = nb_of_label
        self.model = Sequential()
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError
    
    def fit(self):
        self.history = self.model.fit(  x=self.X_train, y=self.y_train,
                                validation_data=(self.X_val, self.y_val),
                                batch_size=self.batch_size,
                                epochs=self.epochs)

    def load_model(self,path):
        self.model = load_model(path)
    
    def save(self,path):
        self.model.save(path)

    def evaluate(self):
        print('Model evaluation:')
        self.model.evaluate(x=self.X_test, y=self.y_test,
                            batch_size=self.batch_size)

    def baseline_result(self):
        print("\nRandom baseline result ...")
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


