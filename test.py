print("Starting ...")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msdi import Msdi
from entry import Entry
from dataReader import DataReader
from baseline_model import Baseline
from logging import info, warn, error

if __name__ =='__main__':
    print("Getting data ...")
    data = DataReader()
    X_train, y_train = data.get_train_data()
    print(np.shape(X_train))
    X_val, y_val = data.get_val_data()
    X_test, y_test = data.get_test_data()

    print("Building model ...")
    baseline = Baseline(X_train, X_val, X_test, y_train, y_val, y_test,
                        data.nb_of_label)
    model = Baseline.build_model()
    
    print("Training model ...")
    model.train()

print('Done !')
