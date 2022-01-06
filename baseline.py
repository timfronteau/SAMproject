from keras.models import Sequential
import abc

class Baseline():
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

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError
    

    def train(self):
        self.model.fit( x=self.X_train, y=self.y_train, validation_data=(self.X_val, self.y_val),
                        batch_size=self.batch_size,
                        epochs=self.epochs)

    def load_model(self,path):
        self.model.load_model(path)
    
    def save(self,path):
        self.model.save(path)

    def evaluate(self):
        self.model.evaluate(x=self.X_test, y=self.y_test,
                            batch_size=self.batch_size)


