# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:23:15 2020

@author: SakataWoolley
"""

from joblib import Parallel, delayed
import tqdm
import pandas as pd
pd.options.display.max_columns = None
import librosa
from datetime import datetime
import numpy as np
import pathlib2


import avgn
from avgn.custom_parsing.bengalese_finch_sakata import (
    generate_json_wav_not_mat,
    parse_song_df,
)
from avgn.utils.paths import DATA_DIR


DATASET_ID = 'zebra_finch_sakata'
species = "Taeniopygia guttata"
common_name = "Zebra Finch"

DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID

# DSLOC = avgn.utils.paths.Path('I:/4avishek_UMAP/kfgj')
# DSLOC = avgn.utils.paths.Path(r'I:\4avishek_UMAP\pi159wh19')
# DSLOC = avgn.utils.paths.Path('I:/4avishek_UMAP/khxv')
DSLOC = avgn.utils.paths.Path('I:/ZebraFinchGenerations/Nest5')
DSLOC


### if single folder 
♣WAVLIST = list((DSLOC).expanduser().glob('*.wav'))
len(WAVLIST), WAVLIST[0]

MATLIST = list((DSLOC).expanduser().glob('*.wav.not.mat'))
len(MATLIST), MATLIST[0]

song_df = parse_song_df(WAVLIST,MATLIST)

song_df[:3]

wav_names = np.array([i.name for i in WAVLIST])

### If folder cotains subfolders all of which should be considered

folder_list = list((DSLOC).expanduser().glob('*'))

WAVPATH = []

for idx, folder_path in enumerate(folder_list):
    WAVLIST = list((pathlib2.Path(folder_path.parent,folder_path.parts[-1])).expanduser().glob('*.wav'))
    MATLIST = list((pathlib2.Path(folder_path.parent,folder_path.parts[-1])).expanduser().glob('*.wav.not.mat'))
    song_temp = parse_song_df(WAVLIST,MATLIST)
    if idx == 0:
        song_df = song_temp
        wav_names = np.array([i.name for i in WAVLIST])
        WAVPATH = WAVLIST
    else:
        song_df = pd.concat([song_df, song_temp], axis=0, ignore_index=True)
        wav_names = np.concatenate((wav_names, np.array([i.name for i in WAVLIST])), axis=0)
        WAVPATH = WAVPATH + WAVLIST
    
# song_df = song_df.drop(columns='index')
WAVLIST = WAVPATH

### Back to main program

Parallel(n_jobs=1, verbose=10)(
    delayed(generate_json_wav_not_mat)(row, WAVLIST, wav_names, DT_ID, species, common_name, DATASET_ID)
    for idx, row in tqdm.tqdm(song_df.iterrows(), total=len(song_df))
);

