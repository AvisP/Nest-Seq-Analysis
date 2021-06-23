from avgn.utils.json import  NoIndent, NoIndentEncoder
import librosa
import json
import avgn
from avgn.utils.paths import DATA_DIR, ensure_dir
import os
# import pathlib
import pathlib2
import pandas as pd
from scipy.io import loadmat
import tqdm
import numpy as np
import time
from datetime import datetime
from avgn.utils.audio import load_wav, read_wav
import soundfile as sf


# DATASET_ID = 'bengalese_finch_sakata'

def generate_json_wav(indv, actual_filename, wav_num, song, sr, DT_ID):
    
    wav_duration = len(song) / sr

    wav_stem = indv + "_" + str(wav_num).zfill(4)

    json_out = ( DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON") )
    # json_out = pathlib2.PureWindowsPath(DATA_DIR).joinpath("processed",DATASET_ID,DT_ID,"JSON",(wav_stem+".JSON"))
    # head,tail = os.path.split(json_out)
    # ensure_dir(pathlib.PureWindowsPath(json_out).parents[0])
    ensure_dir(json_out)
    
        
    # wav_out = pathlib2.PureWindowsPath(DATA_DIR).joinpath("processed",DATASET_ID,DT_ID,"WAV",(wav_stem+".WAV"))
    wav_out = ( DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV"))
    # head,tail = os.path.split(wav_out)
    # ensure_dir(pathlib.PureWindowsPath(wav_out).parents[0])
    # print('Wave Out directory')
    ensure_dir(wav_out)

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["species"] = "Lonchura striata domestica"
    json_dict["common_name"] = "Bengalese finch"
    json_dict["original_filename"] = actual_filename
    json_dict["wav_loc"] = wav_out.as_posix()
    # json_dict["noise_loc"] = noise_out.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_num"] = wav_num

    # add syllable information
    json_dict["indvs"] = {
        indv: {"motifs": {"start_times": [0.0], "end_times": [wav_duration]}}
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    print(wav_out)
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(wav_out, y=song, sr=int(sr), norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
    
def parse_song_df(WAVLIST,MATLIST):
    import pandas as pd
    
    print("Expected wav file format structure : wh70bk90_May_07_2020_26053151_Copy.wav")
    
    if len(WAVLIST) == len(MATLIST):
        print('All wav files have corresponding not.mat files')
    elif len(WAVLIST) > len(MATLIST):
        print('Some wav files do not have mat files')
        print('Following not.mat files are missing \n')   
    elif len(WAVLIST) < len(MATLIST):
        print('Some wav files are missing')
        print('Following not.mat files are missing \n')
        
    
    song_df = pd.DataFrame(
        columns=[
            "bird",
            "species",
            "stime",
            "syllables",
            "start_times",
            "end_times",
            "bout_duration",
            "syll_lens",
            "day",
            "wavname",
            "rate",
        ]
    )
    for label_loc in tqdm.tqdm(MATLIST):
        mat = loadmat(label_loc)
        filename_split = label_loc.stem.split(".")[0].split("_")[1:]

        for val in filename_split:
            if val.isnumeric():
                if len(val) == 8:
                    loc_time = val
                else:
                    loc_time = '999999'
        indv = label_loc.stem.split("_")[0]
        syll_lens = np.squeeze(mat["offsets"] - mat["onsets"]) / 1000
        labels = list(np.array(mat["labels"]).flatten()[0])
        start_times = np.array(mat["onsets"]).flatten() / 1000.0
        end_times = np.array(mat["offsets"]).flatten() / 1000.0
        bout_duration = (mat["offsets"][-1][0] - mat["onsets"][0][0]) / 1000
        wavfilename = label_loc.stem.split(".")[0]
        cur_day = '_'.join(filename_split[0:3])
        song_df.loc[len(song_df)] = [
            indv,
            "ZF",
            loc_time,
            labels,
            start_times,
            end_times,
            bout_duration,
            syll_lens,
            cur_day,
            wavfilename,
            int(mat["Fs"]),
        ]
    song_df["NumNote"] = [len(i) for i in song_df.syllables.values]

    song_df["rec_num"] = None
    song_df = song_df.reset_index()
    for bird in np.unique(song_df.bird):
        song_idxs = song_df[song_df.bird.values == bird].sort_values(by="stime").index
        for idxi, idx in enumerate(song_idxs):
            song_df.at[idx, "rec_num"] = idxi
    # song_df["day"] = [
    #     pd.to_datetime(str(i)).strftime("%Y-%m-%d") for i in song_df.stime.values
    # ]
    return song_df



def generate_json_wav_not_mat(row, WAVLIST, wav_names, DT_ID, species, common_name, DATASET_ID):
    """ generates a json and WAV for bengalese finch data in MAT and CBIN format
    """
    wav_file = np.array(WAVLIST)[wav_names == row.wavname+'.wav'][0]
    print(wav_file)
    rate, bout_wav = read_wav(wav_file.as_posix())

    # general json info
    # make json dictionary
    json_dict = {}
    json_dict["species"] = species
    json_dict["common_name"] = common_name
    json_dict["indvs"] = {
        row.bird: {
            "syllables": {
                "start_times": NoIndent(list(row.start_times)),
                "end_times": NoIndent(list(row.end_times)),
                "labels": NoIndent(list(row.syllables)),
            }
        }
    }
    wav_time = time.strftime('%H:%M:%S.{}'.format(int(float(row.stime)%1000)),time.gmtime(float(row.stime)/1000.0))
   # datetime.datetime.strptime(row.day +' '+ wav_time, "%H:%M:%S.%d")
    json_dict["datetime"] = row.day +' '+ wav_time
    # rate and length
    json_dict["samplerate_hz"] = rate
    json_dict["length_s"] = len(bout_wav) / rate
    
    wav_stem = row.bird + "_" + str(row['index']).zfill(4)

    # wav_stem = row.wavname.split("_")[0]

    # output locations
    wav_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")
    ensure_dir(wav_out)
    
    json_dict["wav_loc"] = wav_out.as_posix()
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )
    ensure_dir(json_out)

    # encode json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    avgn.utils.paths.ensure_dir(wav_out)
#     librosa.output.write_wav(
#         wav_out, y=bout_wav.astype("float32"), sr=int(rate), norm=True
#     )
    sf.write(wav_out, bout_wav.astype("float32"), int(rate), 'PCM_24')

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
