# -*- coding: utf-8 -*-
"""

msdi_io

.. date:: 2020-01-04
.. moduleauthor:: Valentin Emiya
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


_msdi_path = 'msdi'  # Change this to configure your path to MSDI dataset


def get_msdi_dataframe(msdi_path=_msdi_path):
    return pd.read_csv(Path(msdi_path) / 'msdi_mapping.csv')


def load_mfcc(entry, msdi_path=_msdi_path):
    x = np.load(Path(msdi_path) / entry['mfcc'])
    return x[entry['msd_track_id']]


def load_img(entry, msdi_path=_msdi_path):
    return plt.imread(Path(msdi_path) / entry['img'])


def load_deep_audio_features(entry, msdi_path=_msdi_path):
    subset_file = 'X_{}_audio_MSD-I.npy'.format(entry['set'])
    x = np.load(Path(msdi_path) / 'deep_features' / subset_file, mmap_mode='r')
    idx = entry['deep_features']
    return x[idx, :]


def get_set(entry):
    return entry['set']


def get_label(entry):
    return entry['genre']


def get_label_list(msdi_path=_msdi_path):
    df = pd.read_csv(Path(msdi_path) / 'labels.csv', header=None)
    return list(df.iloc[:, 0])


if __name__ == '__main__':
    # Exemple d'utilisation
    msdi = get_msdi_dataframe(_msdi_path)
    print('Dataset with {} entries'.format(len(msdi)))
    print('#' * 80)
    print('Labels:', get_label_list())
    print('#' * 80)

    entry_idx = 23456
    one_entry = msdi.loc[entry_idx]
    print('Entry {}:'.format(entry_idx))
    print(one_entry)
    print('#' * 80)
    mfcc = load_mfcc(one_entry, _msdi_path)
    print('MFCC shape:', mfcc.shape)
    img = load_img(one_entry, _msdi_path)
    print('Image shape:', img.shape)
    deep_features = load_deep_audio_features(one_entry, _msdi_path)
    print('Deep features:', deep_features.shape)
    print('Set:', get_set(one_entry))
    print('Genre:', get_label(one_entry))
