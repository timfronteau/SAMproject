from pathlib import Path
import pandas as pd
import numpy as np


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

    def load_mfcc(self, set = 'all'):
        """ set = {'all','train','test','val'} """
        assert type(set) == str, "set argument = 'all', 'train', 'test' or 'val'"
        print("Loading "+ set +" MFCC ...")
        mfcc_idx = self.df.index.values.tolist()
        mfcc_path = Path(self.msdi_path) / self.df['mfcc']
        res = []
        for (idx,p) in zip(mfcc_idx, mfcc_path):
            entry = self.df.loc[idx]
            if entry['set'] == set :
                x = np.load(p)
                a = np.mean(x[entry['msd_track_id']], axis=0)
                res.append(a)
        n = len(res)
        m = np.shape(res[0])        
        res = np.array(res, dtype=float).reshape(n,m[0],m[1],1)
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

    


    