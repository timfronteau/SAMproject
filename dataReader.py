import numpy as np
from msdi import Msdi
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class DataReader():
    def __init__(self, msdi_path='msdi'):
        self.msdi_path = msdi_path
        self.msdi = Msdi()
        self.nb_of_label = self.msdi.nb_of_label
        self.label_list = np.array(self.msdi.label_list)

    def __get_target(self, set, N=None):
        y = self.msdi.get_label(set)[:N]
        y = LabelEncoder().fit_transform(y)
        return to_categorical(y, num_classes=self.nb_of_label)

    def __get_mfcc_feat(self, set, N=None):
        X = self.msdi.load_mfcc(set, N)
        return X

    def __get_deep_feat(self, set):
        X = self.msdi.load_deep_audio_features(set)
        return X

    def __get_img_feat(self, set):
        X = self.msdi.load_img(set)
        return X

    def __get_txt_feat(self, set):
        X = self.msdi.load_txt(set)
        print(np.shape(X))
        return X

    # MFCC features
    def __get_train_mfcc_data(self, N=None):
        return self.__get_mfcc_feat('train', N), self.__get_target('train', N)

    def __get_val_mfcc_data(self, N=None):
        return self.__get_mfcc_feat('val', N), self.__get_target('val', N)

    def __get_test_mfcc_data(self, N=None):
        return self.__get_mfcc_feat('test', N), self.__get_target('test', N)

    # Deep Features
    def __get_train_deep_features(self):
        return self.__get_deep_feat('train'), self.__get_target('train')

    def __get_val_deep_features(self):
        return self.__get_deep_feat('val'), self.__get_target('val')

    def __get_test_deep_features(self):
        return self.__get_deep_feat('test'), self.__get_target('test')

    # Image Features
    def __get_train_img_features(self):
        return self.__get_img_feat('train'), self.__get_target('train')

    def __get_val_img_features(self):
        return self.__get_img_feat('val'), self.__get_target('val')

    def __get_test_img_features(self):
        return self.__get_img_feat('test'), self.__get_target('test')

    # Text Features
    def __get_train_txt_features(self):
        return self.__get_txt_feat('train'), self.__get_target('train')

    def __get_val_txt_features(self):
        return self.__get_txt_feat('val'), self.__get_target('val')

    def __get_test_txt_features(self):
        return self.__get_txt_feat('test'), self.__get_target('test')

    def get_dataset(self, DATA_TYPE:str, NB_SAMPLE):
        if DATA_TYPE == 'MFCC':
            # MFCC features
            dataset_path_baseline = 'dataset/mfcc.npz'
            X_train, y_train = self.__get_train_mfcc_data(N=NB_SAMPLE)
            X_val, y_val = self.__get_val_mfcc_data(N=NB_SAMPLE)
            X_test, y_test = self.__get_test_mfcc_data(N=NB_SAMPLE)
            # save the data to a .npz file
            np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)

        elif DATA_TYPE == 'Audio':
            # Deep features
            dataset_path_baseline = 'dataset/deep.npz'
            X_train, y_train = self.__get_train_deep_features()
            X_val, y_val = self.__get_val_deep_features()
            X_test, y_test = self.__get_test_deep_features()

            # save the data to a .npz file
            np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)

        elif DATA_TYPE == 'Image':
            # Image features
            dataset_path_baseline = 'dataset/img.npz'
            X_train, y_train = self.__get_train_img_features()
            X_val, y_val = self.__get_val_img_features()
            X_test, y_test = self.__get_test_img_features()

            # save the data to a .npz file
            np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)

        elif DATA_TYPE == 'Text':
            # Text features
            dataset_path_baseline = 'dataset/txt.npz'
            X_train, y_train = self.__get_train_txt_features()
            X_val, y_val = self.__get_val_txt_features()
            X_test, y_test = self.__get_test_txt_features()

            # save the data to a .npz file
            np.savez(dataset_path_baseline, X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)
        else:
            print(f"Invalid argument for {DATA_TYPE}")
            raise ValueError

        return X_train, X_val, X_test, y_train, y_val, y_test