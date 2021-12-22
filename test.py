print("Starting ...")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msdi import Msdi
from entry import Entry
from dataReader import DataReader
from baseline_model import Baseline

if __name__ =='__main__':
    print("Getting data ...")
    data = DataReader()

    # MFCC features
    dataset_path_baseline = 'baseline_mfcc.npz'
    """
    X_train, y_train = data.get_train_data()
    X_val, y_val = data.get_val_data()
    X_test, y_test = data.get_test_data()

    #save the data to a .npz file
    np.savez(dataset_path_baseline,X_train=X_train, X_val=X_val, X_test=X_test,
                                     y_train=y_train, y_val=y_val, y_test=y_test)
    
    """

    # Deep features
    dataset_path_baseline = 'baseline_deep.npz'
    """
    X_train, y_train = data.get_train_deep_features()
    X_val, y_val = data.get_val_deep_features()
    X_test, y_test = data.get_test_deep_features()
    
    #save the data to a .npz file
    np.savez(dataset_path_baseline,X_train=X_train, X_val=X_val, X_test=X_test,
                                     y_train=y_train, y_val=y_val, y_test=y_test)
    """
    
    # Loading data
    print(f"Loading data from {dataset_path_baseline}...")
    dataset_baseline = np.load(dataset_path_baseline)
    X_train = dataset_baseline['X_train']
    X_val = dataset_baseline['X_val']
    X_test = dataset_baseline['X_test']
    y_train = dataset_baseline['y_train']
    y_val = dataset_baseline['y_val']
    y_test = dataset_baseline['y_test']
    

    print("Building model ...")
    baseline = Baseline(X_train, X_val, X_test, y_train, y_val, y_test,
                        data.nb_of_label)
    
    baseline.build_model()
    
    print("Training model ...")
    for k in range(9):
        print(f"Iteration {k}/100")
        try :
            baseline.load_model('deep_model')
        except : None
        baseline.train()
        baseline.evaluate()
        baseline.save('deep_model')


print('Done !')
