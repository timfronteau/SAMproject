import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from utils import mkdir



# Spectrogram generation
def spectrogram_gen(cls):
  img_names = os.listdir('maps_composers/'+cls)
  mkdir('spectrogram/train/'+cls)
  mkdir('spectrogram/test/'+cls)

  lenght_cls = len(img_names)
  train_names = img_names[:int(lenght_cls*2/3)]
  test_names = img_names[int(lenght_cls*2/3):]
  
  
  cnt = 0
  for nm in train_names:
    cnt+=1
    x , _ = librosa.load('maps_composers/'+cls+'/'+nm)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb)
    plt.savefig('spectrogram/train/'+cls+'/'+str(cnt)+'.png')
    plt.close()
  
  cnt = 0
  for nm in test_names:
    cnt+=1
    x , _ = librosa.load('maps_composers/'+cls+'/'+nm)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb)
    plt.savefig('spectrogram/test/'+cls+'/'+str(cnt)+'.png')
    plt.close()

