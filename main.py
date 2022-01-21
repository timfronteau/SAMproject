print("Starting ...")
import numpy as np
from dataReader import DataReader
from baselineImage import BaselineImage
from baselineAudio import BaselineAudio
from baselineText import BaselineText

BATCH_SIZE = 32
EPOCHS = 10
NB_ITERATIONS = 10

GET_AND_SAVE_DATA = False
DATA_TYPE = 'Text'  # 'Audio' 'MFCC' 'Text'

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
    MODEL_CLASS = BaselineAudio  # TODO BaselineMFCC
    MODEL_DIR = 'mfcc_model'
    INPUT_SHAPE = 2048   # TODO CHANGE
else:    # 'Text'
    DATASET_PATH = 'baseline_txt.npz'
    MODEL_CLASS = BaselineText
    MODEL_DIR = 'txt_model'
    INPUT_SHAPE = 5000


if __name__ == '__main__':
    data = DataReader()

    if GET_AND_SAVE_DATA:
        print("Getting data ...")

        if DATA_TYPE == 'MFCC':
            # MFCC features
            dataset_path_baseline = 'baseline_mfcc.npz'

            X_train, y_train = data.get_train_data()
            X_val, y_val = data.get_val_data()
            X_test, y_test = data.get_test_data()

            #save the data to a .npz file
            np.savez(dataset_path_baseline,X_train=X_train, X_val=X_val, X_test=X_test,
                                             y_train=y_train, y_val=y_val, y_test=y_test)

        elif DATA_TYPE == 'Audio':
            # Deep features
            dataset_path_baseline = 'baseline_deep.npz'

            X_train, y_train = data.get_train_deep_features()
            X_val, y_val = data.get_val_deep_features()
            X_test, y_test = data.get_test_deep_features()

            #save the data to a .npz file
            np.savez(dataset_path_baseline,X_train=X_train, X_val=X_val, X_test=X_test,
                                             y_train=y_train, y_val=y_val, y_test=y_test)

        elif DATA_TYPE == 'Image':
            # Image features
            dataset_path_baseline = 'baseline_img.npz'

            X_train, y_train = data.get_train_img_features()
            X_val, y_val = data.get_val_img_features()
            X_test, y_test = data.get_test_img_features()

            # save the data to a .npz file
            np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                     y_train=y_train, y_val=y_val, y_test=y_test)

        else:
            # Text features
            dataset_path_baseline = 'baseline_txt.npz'

            X_train0, y_train0 = data.get_train_txt_features()
            X_val0, y_val0 = data.get_val_txt_features()
            X_test0, y_test0 = data.get_test_txt_features()

            notnan_train = [type(i)==np.ndarray for i in X_train0]
            notnan_val = [type(i) == np.ndarray for i in X_val0]
            notnan_test = [type(i) == np.ndarray for i in X_test0]

            X_train1, y_train1 = np.array(X_train0, dtype=object)[notnan_train], np.array(y_train0, dtype=object)[notnan_train]
            X_val1, y_val1 = np.array(X_val0, dtype=object)[notnan_val], np.array(y_val0, dtype=object)[notnan_val]
            X_test1, y_test1 = np.array(X_test0, dtype=object)[notnan_test], np.array(y_test0, dtype=object)[notnan_test]

            X_train, y_train = np.array(X_train1.tolist()), np.array(y_train1.tolist())
            X_val, y_val = np.array(X_val1.tolist()), np.array(y_val1.tolist())
            X_test, y_test = np.array(X_test1.tolist()), np.array(y_test1.tolist())

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

    baseline = MODEL_CLASS(X_train, X_val, X_test, y_train, y_val, y_test,
                        data.nb_of_label, batch_size=BATCH_SIZE, epochs=EPOCHS, input_shape=INPUT_SHAPE)
        
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
