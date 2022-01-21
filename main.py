print("Starting ...")
import numpy as np
from dataReader import DataReader
from baselineImage import BaselineImage
from baselineAudio import BaselineAudio
from baselineText import BaselineText
from baselineMFCC import BaselineMFCC
from baselineFusion import BaselineFusion


BATCH_SIZE = 32
EPOCHS = 5
NB_ITERATIONS = 2
NB_SAMPLE = None #integer or None value for all the dataset

GET_AND_SAVE_DATA = True
DATA_TYPE = 'Text'  # 'Image' 'Audio' 'MFCC' 'Text' 'Fusion'


if DATA_TYPE=='Image':
    DATASET_PATH = ['baseline_img.npz']
    MODEL_CLASS = BaselineImage
    MODEL_DIR = 'img_model'
    INPUT_SHAPE = (200, 200, 3)
elif DATA_TYPE=='Audio':
    DATASET_PATH = ['baseline_deep.npz']
    MODEL_CLASS = BaselineAudio
    MODEL_DIR = 'deep_model'
    INPUT_SHAPE = 2048
elif DATA_TYPE=='MFCC':
    DATASET_PATH = ['baseline_mfcc.npz']
    MODEL_CLASS = BaselineMFCC
    MODEL_DIR = 'mfcc_model'
    INPUT_SHAPE = 12*3
elif DATA_TYPE=='Text':
    DATASET_PATH = ['baseline_txt.npz']
    MODEL_CLASS = BaselineText
    MODEL_DIR = 'txt_model'
    INPUT_SHAPE = 5000
elif DATA_TYPE=='Fusion':
    DATASET_PATH = ['baseline_img.npz','baseline_deep.npz','baseline_mfcc.npz']
    MODEL_CLASS = BaselineFusion
    MODEL_DIR = 'fusion_model'
    INPUT_SHAPE = [(200, 200, 3), 2048, 12*3]
    
else : print(f"Unvalid argument for DATA_TYPE")

if __name__ == '__main__':
    data = DataReader()
    if GET_AND_SAVE_DATA:
        print("Getting data ...")

        if DATA_TYPE == 'MFCC':
            # MFCC features
            dataset_path_baseline = 'baseline_mfcc.npz'

            X_train, y_train = data.get_train_mfcc_data(N=NB_SAMPLE)
            X_val, y_val = data.get_val_mfcc_data(N=NB_SAMPLE)
            X_test, y_test = data.get_test_mfcc_data(N=NB_SAMPLE)
            
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

        elif DATA_TYPE=='Text':
            # Text features
            dataset_path_baseline = 'baseline_txt.npz'

            X_train, y_train = data.get_train_txt_features()
            X_val, y_val = data.get_val_txt_features()
            X_test, y_test = data.get_test_txt_features()

            # save the data to a .npz file
            np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                     y_train=y_train, y_val=y_val, y_test=y_test)
        else :
            print(f"Unvalid argument for DATA_TYPE")

    # Loading data    
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []
    for path in DATASET_PATH :
        print(f"Loading data from {path}...")
        dataset_baseline = np.load(path, allow_pickle=True)
        X_train.append(dataset_baseline['X_train'])
        X_val.append(dataset_baseline['X_val'])
        X_test.append(dataset_baseline['X_test'])
        y_train.append(dataset_baseline['y_train'])
        y_val.append(dataset_baseline['y_val'])
        y_test = dataset_baseline['y_test']
    print(X_train)

    print("Building model ...")

    baseline = MODEL_CLASS(X_train, X_val, X_test, y_train, y_val, y_test,
                        data.nb_of_label, batch_size=BATCH_SIZE, epochs=EPOCHS, input_shape=INPUT_SHAPE)
        
    baseline.build_model()
    
    baseline.baseline_result()

    print("\nTraining model ...")
    baseline.save_config_to_csv(MODEL_DIR, BATCH_SIZE, EPOCHS,NB_ITERATIONS, DATA_TYPE, DATASET_PATH, MODEL_CLASS, INPUT_SHAPE)

    for k in range(NB_ITERATIONS):
        print(f"Iteration {k}/{NB_ITERATIONS}")
        try:baseline.load_model(MODEL_DIR)
        except: print(f'Aucun model trouvé, un nouveau model sera sauvegardé sous le nom de : {MODEL_DIR}')
        history = baseline.fit()
        baseline.save_history_to_csv(MODEL_DIR, ITERATION=k)
        baseline.evaluate()
        baseline.save(MODEL_DIR)


print('Done !')
