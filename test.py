print("Starting ...")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msdi import Msdi
from entry import Entry
from dataReader import DataReader
from baseline_model import Baseline
import re

BATCH_SIZE = 32
EPOCHS = 10
NB_ITERATIONS = 10

GET_AND_SAVE_DATA = False
DATASET_PATH = 'baseline_img.npz'
MODEL_DIR = 'img_model_conv2'  # 'deep_model'
INPUT_SHAPE = (200, 200, 3) # 2048

if __name__ == '__main__':
    print("Getting data ...")
    data = DataReader()

    if GET_AND_SAVE_DATA:

        # MFCC features
        dataset_path_baseline = 'baseline_mfcc.npz'

        X_train, y_train = data.get_train_data()
        X_val, y_val = data.get_val_data()
        X_test, y_test = data.get_test_data()

        #save the data to a .npz file
        np.savez(dataset_path_baseline,X_train=X_train, X_val=X_val, X_test=X_test,
                                         y_train=y_train, y_val=y_val, y_test=y_test)


        # Deep features
        dataset_path_baseline = 'baseline_deep.npz'

        X_train, y_train = data.get_train_deep_features()
        X_val, y_val = data.get_val_deep_features()
        X_test, y_test = data.get_test_deep_features()

        #save the data to a .npz file
        np.savez(dataset_path_baseline,X_train=X_train, X_val=X_val, X_test=X_test,
                                         y_train=y_train, y_val=y_val, y_test=y_test)

        # Image features
        dataset_path_baseline = 'baseline_img.npz'

        X_train, y_train = data.get_train_img_features()
        X_val, y_val = data.get_val_img_features()
        X_test, y_test = data.get_test_img_features()

        # save the data to a .npz file
        np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                 y_train=y_train, y_val=y_val, y_test=y_test)

    # Loading data
    print(f"Loading data from {DATASET_PATH}...")
    dataset_baseline = np.load(DATASET_PATH)
    X_train = dataset_baseline['X_train']
    X_val = dataset_baseline['X_val']
    X_test = dataset_baseline['X_test']
    y_train = dataset_baseline['y_train']
    y_val = dataset_baseline['y_val']
    y_test = dataset_baseline['y_test']
    

    print("Building model ...")
    baseline = Baseline(X_train, X_val, X_test, y_train, y_val, y_test,
                        data.nb_of_label, batch_size=BATCH_SIZE, epochs=EPOCHS, input_shape=INPUT_SHAPE)

    if not re.findall('conv', MODEL_DIR):
        baseline.build_conv_model()
    else:
        baseline.build_model()
    
    print("Training model ...")
    for k in range(NB_ITERATIONS):
        print(f"Iteration {k}/{NB_ITERATIONS}")
        try:
            baseline.load_model(MODEL_DIR)
        except: pass
        baseline.train()
        baseline.evaluate()
        baseline.save(MODEL_DIR)


print('Done !')
