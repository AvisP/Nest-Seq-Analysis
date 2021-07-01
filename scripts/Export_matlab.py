# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:29:14 2020

@author: SakataWoolley
"""
import pandas as pd
import numpy as np
from scipy.io import savemat

syllable_df = pd.read_pickle(r'I:\avgn_paper-vizmerge\data\syllable_dfs\zebra_finch_sakata\zf_Nest3_noRescale.pickle')
syllable_df.reset_index(drop=True, inplace=True)

indv_list = []
key_list = []
audio_list = []

for i in np.arange(syllable_df.shape[0]):
    indv_list.append(str(syllable_df['indv'][i]))
    key_list.append(str(syllable_df['key'][i]))
    audio_list.append(list(syllable_df['audio'][i]))
    
mdic = {"indv":indv_list, 'key':key_list, 'audio':audio_list}

savemat(r'I:\avgn_paper-vizmerge\Nest3_from_python.mat', mdic)