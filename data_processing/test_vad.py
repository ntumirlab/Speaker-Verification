from Helper import Helper
import os
from glob import glob, iglob
from python_speech_features import fbank, delta, mfcc
import numpy as np
from sklearn import preprocessing
import python_speech_features
import scipy.io.wavfile
import librosa
import DeepSpeaker.Helper
import bob.measure

def preproces_features(input_features):
    # Normalize features with cmvn.

    def normalize_features(clients, m, v):
        for path, feature in clients.items():
            np.save(path, ((feature - m) / v))

        return clients

    features = np.vstack(feature for feature in input_features.values())
    scaler = preprocessing.StandardScaler().fit(features)
    input_features = normalize_features(input_features, scaler.mean_, scaler.var_)
    return input_features



if __name__ == "__main__":



    wav_path = f"/mnt/E/arthur.wang/Google_speech_command/wav/bed/0a7c2a8d_nohash_0.wav"

    sr, audio = scipy.io.wavfile.read(wav_path)

    print(audio)
    print(sr)
    vad = bob.bio.spear.preprocessor.Energy_Thr(
        smoothing_window = 10, \
        win_length_ms = 25, \
        win_shift_ms = 10, \
        ratio_threshold = 0.1
    )   
    _, _, labels = vad([sr, audio.astype('float')])
    # if audio.dtype == 'int16':
    #     nb_bits = 16 # -> 16-bit wav files
    # elif audio.dtype == 'int32':
    #     nb_bits = 32 # -> 32-bit wav files
    # max_nb_bit = float(2 ** (nb_bits - 1))
    # audio = audio / (max_nb_bit + 1.0)
