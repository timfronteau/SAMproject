import numpy as np 

def extract_config(DATA_TYPE:str, MULTI_TYPE:str):
    if MULTI_TYPE == 'Uni':
        if DATA_TYPE == 'Image':
            from imageModel import ImageModel
            DATASET_PATH = ['dataset/img.npz']
            MODEL_CLASS = ImageModel
            MODEL_DIR = 'saved_model/img_model'
            INPUT_SHAPE = (200, 200, 3)
        elif DATA_TYPE == 'Audio':
            from audioModel import AudioModel
            DATASET_PATH = ['dataset/deep.npz']
            MODEL_CLASS = AudioModel
            MODEL_DIR = 'saved_model/deep_model'
            INPUT_SHAPE = 2048
        elif DATA_TYPE == 'MFCC':
            from MFCCModel import MFCCModel
            DATASET_PATH = ['dataset/mfcc.npz']
            MODEL_CLASS = MFCCModel
            MODEL_DIR = 'saved_model/mfcc_model'
            INPUT_SHAPE = 12*3
        elif DATA_TYPE == 'Text':
            from textModel import TextModel
            DATASET_PATH = ['dataset/txt.npz']
            MODEL_CLASS = TextModel
            MODEL_DIR = 'saved_model/txt_model'
            INPUT_SHAPE = 5000
    elif DATA_TYPE == 'All':
        DATASET_PATH = ['dataset/img.npz', 'dataset/deep.npz', 'dataset/mfcc.npz', 'dataset/txt.npz']
        if MULTI_TYPE == 'Late':
            from lateModel import LateModel
            MODEL_CLASS = LateModel
            MODEL_DIR = 'saved_model/late_model'
            INPUT_SHAPE = [(200, 200, 3), 2048, 12 * 3, 5000]
        elif MULTI_TYPE == 'Hybrid':
            from hybridModel import HybridModel
            MODEL_CLASS = HybridModel
            MODEL_DIR = 'saved_model/hybrid_model'
            INPUT_SHAPE = [(200, 200, 3), 2048, 12 * 3, 5000]
        elif MULTI_TYPE == 'Early':
            from earlyModel import EarlyModel
            MODEL_CLASS = EarlyModel
            MODEL_DIR = 'saved_model/early_model'
            INPUT_SHAPE = [(200, 200, 3), 2048, 12 * 3, 5000]
        else:
            print(f"{MULTI_TYPE} not implemented")

    else:
        print(f"Invalid argument for {DATA_TYPE}")
        raise ValueError

    return DATASET_PATH, MODEL_CLASS, MODEL_DIR,  INPUT_SHAPE


def get_dataset_from_file(DATASET_PATH:list):
        X_train, X_val, X_test = [], [], []
        for path in DATASET_PATH:
            print(f"Loading data from {path}...")
            dataset_baseline = np.load(path, allow_pickle=True)
            X_train.append(dataset_baseline['X_train'])
            X_val.append(dataset_baseline['X_val'])
            X_test.append(dataset_baseline['X_test'])

        y_train = dataset_baseline['y_train']
        y_val = dataset_baseline['y_val']
        y_test = dataset_baseline['y_test']
        return X_train, X_val, X_test, y_train, y_val, y_test