# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:56:59 2020

@author: SakataWoolley
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
#import umap
import pandas as pd

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
from avgn.utils.hparams import HParams
from avgn.dataset import DataSet

from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav, create_label_df, get_row_audio
from avgn.visualization.spectrogram import plot_spec
from avgn.song_segmentation.dynamic_thresholding import norm

DATASET_ID = 'zebra_finch_sakata'

hparams = HParams(
    n_fft = 4096,
    mel_lower_edge_hertz=500,
    mel_upper_edge_hertz=15000,  # Should be sample_rate / 2 or less
    butter_lowcut = 500,
    butter_highcut = 15000,
    ref_level_db = 20,
    min_level_db = -100,
    win_length_ms = 10,
    hop_length_ms = 1,
    num_mel_bins = 32,
    mask_spec = True,
    n_jobs = 1,  # Makes processing serial if set to 1, parallel processing giving errors
    verbosity=1,
    nex = -1
)

# sub_dir = '2021-01-03_02-07-49'
# dataset = DataSet(DATASET_ID, hparams = hparams, sub_dir = sub_dir)

dataset = DataSet(DATASET_ID, hparams = hparams)

dataset.sample_json

len(dataset.data_files)

# plot_spec(
#     norm(spec),
#     fig=None,
#     ax=None,
#     rate=None,
#     hop_len_ms=None,
#     cmap=plt.cm.afmhot,
#     show_cbar=True,
#     figsize=(20, 6),
# )


### Create dataset based upon JSON

from joblib import Parallel, delayed
n_jobs = 1; verbosity = 10

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(create_label_df)(
            dataset.data_files[key].data,
            hparams=dataset.hparams,
            labels_to_retain=['labels'],
            unit="syllables",
            dict_features_to_retain = [],
            key = key,
        )
        for key in tqdm(dataset.data_files.keys())
    )
syllable_df = pd.concat(syllable_dfs)
len(syllable_df)

## Add wav location and audio to dataframe

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(get_row_audio)(
            syllable_df[syllable_df.key == key], 
            dataset.data_files[key].data['wav_loc'], 
            dataset.hparams
        )
        for key in tqdm(syllable_df.key.unique())
    )
syllable_df = pd.concat(syllable_dfs)
len(syllable_df)

# Check for bad audio files
df_mask  = np.array([len(i) > 0 for i in tqdm(syllable_df.audio.values)])
syllable_df = syllable_df[np.array(df_mask)]

# Plot some audio syllables
sylls = syllable_df.audio.values

nrows = 5
ncols = 10
zoom = 2
fig, axs = plt.subplots(ncols=ncols, nrows = nrows,figsize = (ncols*zoom, nrows+zoom/1.5))
for i, syll in tqdm(enumerate(sylls), total = nrows*ncols):
    ax = axs.flatten()[i]
    ax.plot(syll)
    if i == nrows*ncols -1:
        break

    
# Make spectrogram
from avgn.visualization.spectrogram import draw_spec_set
from avgn.visualization.spectrogram import draw_spec_set_actual
from avgn.signalprocessing.create_spectrogram_dataset import make_spec, mask_spec, log_resize_spec, pad_spectrogram
from avgn.visualization.spectrogram import plot_spec, visualize_spec

from avgn.utils.audio import load_wav, read_wav
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.signalprocessing.spectrogramming import spectrogram

#### -----------   Plotting a sample data file --------------------- ####
# mypath = r'I:\avgn_paper-vizmerge\data\processed\zebra_finch_sakata\2020-06-11_18-46-49\WAV'
# file_current = 'wh70bk90_0000.WAV'
# mypath = r'I:\avgn_paper-vizmerge\data\processed\zebra_finch_sakata2\2020-06-08_13-24-48\WAV'
# file_current = 'pi159wh19_0000.WAV'

# mypath = r'I:\avgn_paper-vizmerge\data\processed\zebra_finch_sakata\2020-06-19_17-02-39\WAV'
# file_current = 'kfgj_0000.WAV'

# mypath = r'I:\avgn_paper-vizmerge\data\processed\zebra_finch_sakata\2020-06-19_18-55-22\WAV'
# file_current = 'khxv_0000.WAV'

(dataset.wav_files[10].as_posix())

# rate, data_loaded = load_wav(mypath+'\\'+file_current)
rate, data_loaded = load_wav(dataset.wav_files[49])
data = data_loaded
times = np.linspace(0,len(data)/rate,len(data));


# filter data
butter_min = hparams.butter_lowcut
butter_max = hparams.butter_highcut
data = butter_bandpass_filter(data, butter_min, butter_max, rate)

fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(10, 6))
axs[0].plot(times,data)

hparams.sample_rate = rate

hparams.ref_level_db = 60
spec_orig = spectrogram(data,
                            rate,
                            hparams)
plot_spec(
    norm(spec_orig),
    fig=fig,
    ax=axs[1],
    rate=rate,
    hop_len_ms=hparams.hop_length_ms,
    cmap=plt.cm.afmhot,
    show_cbar=False,
    figsize=(10, 6),
)

# plot_spec(
#     norm(spec_orig[:1024,:]),
#     fig=None,
#     ax=None,
#     rate=rate,
#     hop_len_ms=hparams.hop_length_ms,
#     cmap=plt.cm.afmhot,
#     show_cbar=False,
#     figsize=(10, 6),
# )

###############   -----------------------------------   ########

syllables_wav = syllable_df.audio.values
syllables_rate = syllable_df.rate.values

###  ---  Indivdual audio snippet  ----   ###
i=1265

spec = spectrogram(syllables_wav[i], syllables_rate[i], hparams)

times = np.linspace(0,len(syllables_wav[i])/syllables_rate[i],len(syllables_wav[i]));

fig, axs = plt.subplots(nrows=2,ncols=1, figsize=(5, 6))
axs[0].plot(times,syllables_wav[i])

plot_spec(
    norm(spec),
    fig=fig,
    ax=axs[1],
    rate=syllables_rate[i],
    hop_len_ms=hparams.hop_length_ms,
    cmap=plt.cm.afmhot,
    show_cbar=False,
    figsize=(5, 6),
)

### --- Displays specs in sequence of windows --- ###

for i in range(0,50):
    fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(5, 6))
    plot_spec(
        norm(syllable_df.spectrogram.values[i]),
        fig=fig,
        ax=axs,
        rate=syllables_rate[i],
        hop_len_ms=hparams.hop_length_ms,
        cmap=plt.cm.afmhot,
        show_cbar=False,
        figsize=(5, 6),
    )
    plt.title(i)
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    
### --- Compute Spectrograms --- ###

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    # create spectrograms
    syllables_spec = parallel(
        delayed(spectrogram)(
            syllable,
            rate,
            hparams=dataset.hparams
        #     mel_matrix=dataset.mel_matrix,
        #     use_mel=True,
        #     use_tensorflow=False,
         )
        for syllable, rate in tqdm(
            zip(syllables_wav, syllables_rate),
            total=len(syllables_rate),
            desc="getting syllable spectrograms",
            leave=False,
        )
    )
# draw_spec_set(spectrograms, maxrows=3, colsize=10, cmap=plt.cm.afmhot, zoom=2,fig_title='unknown',num_indv='unspecified'):

# with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
#     # create spectrograms
#     syllables_spec = parallel(
#         delayed(make_spec)(
#             syllable,
#             rate,
#             hparams=dataset.hparams,
#             mel_matrix=dataset.mel_matrix,I had a 
#             use_mel=True,
#             use_tensorflow=False,
#         )
#         for syllable, rate in tqdm(
#             zip(syllables_wav, syllables_rate),
#             total=len(syllables_rate),
#             desc="getting syllable spectrograms",
#             leave=False,
#         )
#     )
    
####  -----------  Rescaling --------  #### (Avoid this if possible)
    
log_scaling_factor = 4

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
        for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
    )  
    
# draw_spec_set_actual(syllables_spec, zoom=1, maxrows=10, colsize=25)

#### --- Pad Spectrograms ---- ###
syll_lens = [np.shape(i)[1] for i in syllables_spec]
pad_length = np.max(syll_lens) # Set this as 467 based on Nest 2

# import seaborn as sns
# for indv in np.unique(syllable_df.indv):
#     sns.distplot(np.log(syllable_df[syllable_df.indv==indv]["end_time"] - syllable_df[syllable_df.indv==indv]["start_time"]), label=indv)
# plt.legend()

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:

    syllables_spec = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm(
            syllables_spec, desc="padding spectrograms", leave=False
        )
    )
np.shape(syllables_spec)

###  Discards the top half of frequency range ###

def freq_range_reduction(spec, yrange=0.5):
    ylim = int(np.floor(spec.shape[0]*yrange))
    spec = spec[:ylim,:]
    return spec

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    
    syllables_spec = parallel(
        delayed(freq_range_reduction)(spec, yrange=0.5)
        for spec in tqdm(
            syllables_spec, desc="resizing spectrogram frequency", leave=False
        )
    )
np.shape(syllables_spec)

### To save space back to dataframe
syllables_spec = [(norm(i)*255).astype('uint8') for i in tqdm(syllables_spec)]

syllable_df['spectrogram'] = syllables_spec

# save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf_pi159wh19.pickle'
# save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf_wh70bk90.pickle'

# save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf_kfgj.pickle'
# save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf_khxv.pickle'
save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf_Nest5_noRescale.pickle'
ensure_dir(save_loc)
syllable_df.to_pickle(save_loc)

# with open('syllables_spec.pickle', 'wb') as handle:
#     pickle.dump(syllables_spec, handle, protocol=pickle.HIGHEST_PROTOCOL)
