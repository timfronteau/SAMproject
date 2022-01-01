
import numpy as np
from msdi import Msdi
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class DataReader():
    def __init__(self, msdi_path = 'msdi'):
        self.msdi_path = msdi_path
        self.msdi = Msdi()
        self.nb_of_label = self.msdi.nb_of_label
        self.label_list = np.array(self.msdi.get_label_list)

    def __get_target(self,set):        
        y = self.msdi.get_label(set)
        y = LabelEncoder().fit_transform(y)
        return to_categorical(y, num_classes=self.nb_of_label)
        #return np.array([1*([y[i]]*self.nb_of_label == self.label_list) for i in range(len(y))],
        #                dtype=int)


    def __get_feat(self, set):
        X = self.msdi.load_mfcc(set)
        X = np.array([np.mean(x, axis=0) for x in X], dtype=float)
        X = X.reshape([len(X),12])   
        return X

    def __get_deep_feat(self,set):
        X = self.msdi.load_deep_audio_features(set)
        print(np.shape(X))
        return X

    def __get_img_feat(self,set):
        #Done
        X = self.msdi.load_img(set)
        print(np.shape(X))
        return X

    # MFCC features
    def get_train_data(self):
        return self.__get_feat('train'), self.__get_target('train')  
        
    def get_val_data(self):
        return self.__get_feat('val'), self.__get_target('val')  
    
    def get_test_data(self):
        return self.__get_feat('test'), self.__get_target('test')  
    
    # Deep Features
    def get_train_deep_features(self):
        return self.__get_deep_feat('train'), self.__get_target('train')  
        
    def get_val_deep_features(self):
        return self.__get_deep_feat('val'), self.__get_target('val')  
    
    def get_test_deep_features(self):
        return self.__get_deep_feat('test'), self.__get_target('test')

    # Image Features
    def get_train_img_features(self):
        #Done
        return self.__get_img_feat('train'), self.__get_target('train')

    def get_val_img_features(self):
        #Done
        return self.__get_img_feat('val'), self.__get_target('val')

    def get_test_img_features(self):
        #Done
        return self.__get_img_feat('test'), self.__get_target('test')