import os
import librosa
import matplotlib.pyplot as plt
from utils import mkdir

# Wavelet data generation
def wavelet_data_gen(cls):
  img_names = os.listdir('maps_composers/'+cls)
  mkdir('wavelets/train/'+cls)
  mkdir('wavelets/test/'+cls)

  lenght_cls = len(img_names)
  train_names = img_names[:int(lenght_cls*2/3)]
  test_names = img_names[int(lenght_cls*2/3):]
  cnt = 0
  for nm in train_names:
    cnt+=1
    x , _ = librosa.load('maps_composers/'+cls+'/'+nm)
    #plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x)
    plt.savefig('wavelets/train/'+cls+'/'+str(cnt)+'.png')
    plt.close()
  
  cnt = 0
  for nm in test_names:
    cnt+=1
    x , _ = librosa.load('maps_composers/'+cls+'/'+nm)
    #plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x)
    plt.savefig('wavelets/test/'+cls+'/'+str(cnt)+'.png')
    plt.close()

