from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.models import Sequential, Model
from keras import Input
from keras.applications.vgg19 import VGG19

from model import Model


class ImageModel(Model):
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=16, epochs=10, input_shape=(200, 200, 3)):
        super().__init__(X_train, X_val, X_test, y_train, y_val, y_test, nb_of_label, batch_size=batch_size, epochs=epochs, input_shape=input_shape)
        self.model_name = 'model_Image'
        
    def build_model(self,type='simple'):
        if type=='simple': self.__build_model_simple()
        elif type=='transfert': self.__build_model_transfert()
        
    def __build_model_simple(self):
        m = Sequential(name='model_image')
        m.add(Input(shape=self.input_shape, name='image_input'))
        m.add(MaxPooling2D((4, 4), name='image_maxpool_1'))
        m.add(Conv2D(16, (3, 3), activation='relu', name='image_conv2d_1'))
        m.add(MaxPooling2D((2, 2), name='image_maxpool_2'))
        m.add(Conv2D(32, (3, 3), activation='relu', name='image_conv2d_2'))
        m.add(MaxPooling2D((2, 2), name='image_maxpool_3'))
        m.add(Conv2D(32, (3, 3), activation='relu', name='image_conv2d_3'))
        size = int((int((int(self.input_shape[0]/4)-2)/2)-2)/2)
        m.add(AveragePooling2D((size, size), padding='same', name='image_avgpool_1'))
        m.add(Flatten(name='image_flatten'))
        m.add(BatchNormalization(name='image_batchnorm'))
        m.add(Dropout(0.1, name='image_dropout_1'))
        m.add(Dense(100, activation='relu', name='image_dense_1'))
        m.add(Dropout(0.1, name='image_dropout_2'))
        m.add(Dense(100, activation='relu', name='image_dense_2'))
        m.add(Dropout(0.1, name='image_dropout_3'))
        m.add(Dense(self.nb_of_label, activation='relu', name='image_dense_3'))
        m.add(Dense(self.nb_of_label, activation='softmax', name='image_output'))

        m.compile(loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = m

    def __build_model_transfert(self):
        vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        vgg19 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_pool').output)
        vgg19.trainable = False
        #Building a new network to plug after the first one

        x = Input(shape=(6,6,512,), name='input')
        y0 = BatchNormalization()(x)
        y0 = AveragePooling2D((14,14), padding='same')(x)
        y0 = Flatten()(y0)
        y0 = Dropout(0.1)(y0)
        y0 = Dense(128, activation='relu')(y0)
        y0 = Dropout(0.1)(y0)
        y0 = Dense(32, activation='relu')(y0)
        y0 = Dropout(0.1)(y0)
        y0 = Dense(self.nb_of_label, activation='relu')(y0)
        y0 = Dense(self.nb_of_label, activation='softmax')(y0)
        submodel_dense = Model(inputs=x, outputs=y0)
        submodel_dense.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


        #Combine the networks
        m = Model(inputs=vgg19.input, outputs=submodel_dense(vgg19.output))
        m.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.model = m