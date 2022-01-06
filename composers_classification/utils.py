import os
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cv2

def get_confusion_matrix(y_val, predictions, labels, path):
  cm1 = confusion_matrix(y_val, predictions)
  df_cm = pd.DataFrame(cm1, index = [i for i in labels],
                columns = [i for i in labels])
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm, annot=True,cmap="RdPu")
  plt.savefig(path,bbox_inches = 'tight')

def get_sim_plot(EPOCHS, SIM_TYPE):
  history = pickle.load(open(f'history_{EPOCHS}_epoch_{SIM_TYPE}.pkl','rb'))
  acc = history['accuracy']
  val_acc = history['val_accuracy']
  loss = history['loss']
  val_loss = history['val_loss']

  epochs_range = range(EPOCHS)
  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
  plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=10)
  ax1.plot(epochs_range, acc, label='Training Accuracy', c = '#4CAF50', linewidth=4)
  ax1.plot(epochs_range, val_acc, label='Validation Accuracy', c='red', linewidth=4)
  ax1.legend()
  ax1.set_title('Training and Validation Accuracy',fontsize=18)
  ax1.set_ylabel('Accuracy',fontsize=18)
  ax1.set_xlabel('Epoch',fontsize=18)

  ax2.plot(epochs_range, loss, label='Training Loss',c = '#4CAF50', linewidth=4)
  ax2.plot(epochs_range, val_loss, label='Validation Loss', c='red', linewidth=4)
  ax2.legend()
  ax2.set_title('Training and Validation Loss',fontsize=18)
  ax2.set_ylabel('Loss',fontsize=18)
  ax2.set_xlabel('Epoch',fontsize=18)
  fig.tight_layout(pad=3.0)
  #plt.show()
  plt.savefig(f'sim_plot_{EPOCHS}_epoch_{SIM_TYPE}.png',bbox_inches = 'tight')
  plt.clf()


def mkdir(dir):
  if os.path.exists(dir):
    shutil.rmtree(dir)
  os.makedirs(dir)

def get_data(data_dir, labels):
  data = [] 
  for label in labels: 
      path = os.path.join(data_dir, label)
      class_num = labels.index(label)
      for img in os.listdir(path):
          try:
              img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
              resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
              data.append([resized_arr, class_num])
          except Exception as e:
              print(e)
  return np.array(data)