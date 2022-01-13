from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys

class Msdi():
    def __init__(self, msdi_path = 'msdi'):
        self.msdi_path = msdi_path
        self.df = self.get_msdi_dataframe()
        self.nb_of_entry = self.get_nb_of_entry()
        self.label_list = self.get_label_list()
        self.nb_of_label = len(self.label_list)
        
    def get_msdi_dataframe(self):
        return pd.read_csv(Path(self.msdi_path) / 'msdi_mapping.csv')

    def get_nb_of_entry(self):
        return len(self.get_msdi_dataframe())

    def get_msdi_path(self):
        return self.msdi_path

    def load_mfcc(self, set = 'all',N=None):
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" MFCC ...")
        dataset = self.df.loc[(self.df['set']==set)]
        res = [np.load(Path(self.msdi_path) / mfcc)[msd_track_id] for (mfcc,msd_track_id) in zip(dataset['mfcc'][:N], dataset['msd_track_id'][:N])]
        res = np.array([[np.mean(x, axis=0),
                        np.mean(librosa.feature.delta(x,order=1),axis=0),
                        np.mean(librosa.feature.delta(x,order=2),axis=0)] for x in res], dtype=float)
        res = res.reshape([len(res),12*3]) 
        return res

    def load_deep_audio_features(self, set):
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" deep audio features ...")
        subset_file = 'X_{}_audio_MSD-I.npy'.format(set)
        x = np.load(Path(self.msdi_path) / 'deep_features' / subset_file, mmap_mode='r')
        deep_features_idx = self.df.loc[(self.df['set']==set)]['deep_features'].tolist()
        res = x[deep_features_idx, :]
        return res

    def load_img(self, set = 'all'):
        #Done
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" images ...")
        dataset = self.df.loc[(self.df['set']==set)]
        res = [plt.imread(Path(self.msdi_path) / img) for img in dataset['img']]
        return res

    def load_txt(self, set = 'all'):
        #TODO
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" images ...")
        dataset = self.df.loc[(self.df['set']==set)]
        res = [plt.imread(Path(self.msdi_path) / img) for img in dataset['img']]
        return res

    def get_label(self, set = 'all'):
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" label ...")
        mfcc_idx = self.df.index.values.tolist()
        res = []
        for idx in mfcc_idx:
            entry = self.df.loc[idx]
            if entry['set'] == set :
                res.append(entry['genre'])
        return res
        

    def get_label_list(self):
        df = pd.read_csv(Path(self.msdi_path) / 'labels.csv', header=None)
        return list(df.iloc[:, 0])

    


    