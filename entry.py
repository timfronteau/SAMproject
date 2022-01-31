from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msdi import Msdi

class Entry(Msdi):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.entry = self.get_entry()

    def load_mfcc(self):
        x = np.load(Path(self.msdi_path) / self.entry['mfcc'])
        return x[self.entry['msd_track_id']]

    def load_img(self):
        return plt.imread(Path(self.msdi_path) / self.entry['img'])

    def get_entry(self):
        assert type(self.idx) == int, 'The index idx must be an integer'
        return self.get_msdi_dataframe().loc[self.idx]

    def load_deep_audio_features(self):
        subset_file = 'X_{}_audio_MSD-I.npy'.format(self.entry['set'])
        x = np.load(Path(self.msdi_path) / 'deep_features' / subset_file, mmap_mode='r')
        idx = self.entry['deep_features']
        return x[idx, :]

    def get_nb_of_deep_audio_feat(self):
        return self.load_deep_audio_features().shape

    def get_set(self):
        return self.entry['set']

    def get_label(self):
        return self.entry['genre']

    def __str__(self):
        resultString = []

        resultString.append('Dataset with {} entries'.format(self.get_nb_of_entry()))
        resultString.append('#' * 80)
        resultString.append('Labels:'+str(self.get_label_list()))
        resultString.append('#' * 80)
        
        resultString.append('Entry {}:'.format(self.idx))
        resultString.append(self.entry.to_string())
        resultString.append('#' * 80)
        mfcc = self.load_mfcc()
    
        resultString.append('MFCC shape:'+str(mfcc.shape))
        img = self.load_img()
        resultString.append('Image shape:'+str(img.shape))
        resultString.append('Deep features:'+str(self.get_nb_of_deep_audio_feat()))
        resultString.append('Set:'+str(self.get_set()))
        resultString.append('Genre:'+str(self.get_label()))

        return "\n".join(resultString)