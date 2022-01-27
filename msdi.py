from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

class Msdi():
    def __init__(self, msdi_path = 'msdi'):
        self.msdi_path = msdi_path
        self.df = self.get_msdi_dataframe()
        #self.text_df = self.get_text_dataframe()
        self.nb_of_entry = self.get_nb_of_entry()
        self.label_list = self.get_label_list()
        self.nb_of_label = len(self.label_list)
        
    def get_msdi_dataframe(self):
        df_msdi = pd.read_csv(Path(self.msdi_path) / 'msdi_mapping.csv')

        file = open(f"mxm/msx_lyrics_genre.txt", "r")
        lines = file.readlines()
        ids = []
        text_bow = []
        for i, line in enumerate(lines):
            line_list = line.split()
            ids.append(line_list[0])
            bow = np.zeros(5000)
            for word in line_list[2:]:
                wd, nb = word.split(':')
                bow[int(wd) - 1] = int(nb)
            text_bow.append(bow)
        df_text = pd.DataFrame(data={'msd_track_id': ids,
                                  'bow': text_bow})

        df = pd.merge(df_msdi, df_text, on='msd_track_id', how='left')

        for idx, elt in enumerate(df['bow']):
            if type(elt) != np.ndarray:
                df['bow'].iat[idx] = np.zeros(5000)
        return df

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
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" images ...")
        dataset = self.df.loc[(self.df['set']==set)]
        res = [plt.imread(Path(self.msdi_path) / img) for img in dataset['img']]
        return res

    def load_txt(self, set = 'all'):
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" text BOWs ...")
        dataset = self.df.loc[(self.df['set']==set)]
        res = [bow for bow in dataset['bow']]
        return res

    def get_label(self, set = 'all'):
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" labels ...")
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

    


    