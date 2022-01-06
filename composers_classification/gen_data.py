import os
from wavelet_data_gen import wavelet_data_gen
from spectrogram_gen import spectrogram_gen





def gen_data():
    print("start")
    classes = [a for a in os.listdir('maps_composers') if '.' not in a]
    print(classes)
    for cls in classes :
        print(cls)
        print("Spec wavelet")
        spectrogram_gen(cls)
        print("Gen wavelet")
        wavelet_data_gen(cls)