from msdi import Msdi


class DataReader():
    def __init__(self, msdi_path = 'msdi'):
        self.msdi_path = msdi_path
        self.msdi = Msdi()
        self.nb_of_label = self.msdi.nb_of_label

    def __get_data(self, set):
        X = self.msdi.load_mfcc(set)
        y = self.msdi.get_label(set)
        return X, y

    def get_train_data(self):
        return self.__get_data('train')
        
    def get_val_data(self):
        return self.__get_data('val')
    
    def get_test_data(self):
        return self.__get_data('test')