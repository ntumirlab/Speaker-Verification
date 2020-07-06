from Helper import Helper
import os
from glob import glob, iglob
from python_speech_features import fbank, delta, mfcc
import numpy as np
from sklearn import preprocessing
import python_speech_features
import scipy.io.wavfile
import librosa
import librosa.display
import DeepSpeaker.Helper
import bob.measure
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # wav_path = r"/mnt/E/arthur.wang/Google_speech_command/wav/bed/0a7c2a8d_nohash_0.wav"
    wav_path = r"/mnt/E/arthur.wang/SRE10_WAV/data/phonecall/tel/taaue.wav"
    wav_path = r"left.wav"

    sr, audio = scipy.io.wavfile.read(wav_path)
    print(sr,audio.shape)
    exit()
    channel = 1

    audio = audio[:, int(channel)]
    # scipy.io.wavfile.write('right.wav', sr, audio)
    # scipy.io.wavfile.write('left.wav', sr, audio[:, 0])
    print(len(audio))
    # exit()
    vad = bob.bio.spear.preprocessor.Energy_Thr(
        smoothing_window = 10, \
        win_length_ms = 25, \
        win_shift_ms = 10, \
        ratio_threshold = 0.1
    )   
    _, _, labels = vad([sr, audio.astype('float')])

    # only_right = np.append(audio, audio)


    if audio.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif audio.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    audio = audio / (max_nb_bit + 1.0)
    
    test_wav = []
    for i in range(len(labels)):
        if labels[i] == 1:
            test_wav.append(audio[i])
    test_wav = np.array(test_wav)
    print(len(audio))
    print(len(test_wav))
    scipy.io.wavfile.write('test.wav', sr, test_wav)

    exit()
    # y = np.append(audio, audio)

    librosa.output.write_wav('test.wav', np.transpose(audio), sr, mon)


    exit()
    psf_mfcc = mfcc(audio, samplerate=sr, numcep=39, nfilt=39)

    frame_size = 25
    hop = 10
    bob_mfcc = Helper.get_mfcc(wav_path, int(channel), frame_size, hop, True)

    hop_duration = 0.01 # in seconds
    hop_length = int(sr * hop_duration)
    win_length = 2 * hop_length
    n_fft = win_length
    lib_mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    vad_lib_mfcc = []
    lib_mfcc = np.transpose(lib_mfcc)
    for i in range(len(labels)):
        if labels[i] == 1:
            vad_lib_mfcc.append(lib_mfcc[i+1])
    vad_lib_mfcc = np.transpose(np.array(vad_lib_mfcc))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(np.transpose(psf_mfcc), x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig("psf_google.png")
    plt.cla()
    plt.clf()

    librosa.display.specshow(np.transpose(bob_mfcc), x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig("bob_google.png")
    plt.cla()
    plt.clf()
    librosa.display.specshow(vad_lib_mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig("libro_google.png")
    plt.cla()
    plt.clf()
    cc = 0

    exit()


