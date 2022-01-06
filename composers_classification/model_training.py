import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
import os
import numpy as np
from utils import get_data, get_confusion_matrix, get_sim_plot

labels = [a for a in os.listdir('maps_composers') if '.' not in a]
img_size = 256
EPOCHS = 100
SIM_TYPE = "simple"


train = get_data('spectrogram/train', labels)
val = get_data('spectrogram/test', labels)

x_train = [features for features, _ in train]
y_train = [label for _, label in train]

x_val = [features for features, _ in val]
y_val = [label for _, label in val]

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

# Prepare the data
x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        #rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(256,256,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(4, activation="softmax"))

model.summary()

opt = Adam(lr=0.0001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = EPOCHS, validation_data = (x_val, y_val))

import pickle
model.save_weights(f'{EPOCHS}_epoch_{SIM_TYPE}_lr.cpkt')

pickle.dump(history.history, open(f'history_{EPOCHS}_epoch_{SIM_TYPE}.pkl','wb'))


get_sim_plot(EPOCHS, SIM_TYPE)


predictions = np.argmax(model.predict(x_val), axis=1)
predictions = predictions.reshape((1,-1))[0]
print(classification_report(y_val, predictions, target_names = labels))


get_confusion_matrix(y_val, predictions, labels, path='confusion_mrtx_{SIM_TYPE}.png')
