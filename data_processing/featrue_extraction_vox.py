from glob import iglob
import numpy as np
import scipy.io.wavfile
import librosa
from tqdm import tqdm
import os
import pickle as pkl


if __name__ == "__main__":

    data_dir = "/mnt/E/arthur.wang/vox1/vad_wav"

    f = iglob(f"{data_dir}/*/*/*/*.wav")
    for path in tqdm(f, total=153516):

        tokens = path.split('/')
        filename = tokens[-1]

        # sr, audio = scipy.io.wavfile.read(path)
        audio, sr = librosa.load(path, sr=16000)
        # print(sr, audio)
        # exit()
        # print(audio)
        # if audio.dtype == 'int16':
        #     nb_bits = 16  # -> 16-bit wav files
        # elif audio.dtype == 'int32':
        #     nb_bits = 32  # -> 32-bit wav files


        # max_nb_bit = float(2 ** (nb_bits - 1))
        # audio = audio / (max_nb_bit + 1.0)
        # print(audio)
        # exit()
        audio = librosa.effects.preemphasis(audio)

        hop_duration = 0.01  # in seconds
        win_duration = 0.025
        hop_length = int(sr * hop_duration)
        n_fft = int(sr * win_duration)

        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39, n_fft=n_fft, hop_length=hop_length)
            fbank = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64)
            logfbank = librosa.power_to_db(fbank)
        except:
            continue

        output_dir = path.replace(f"/{filename}", "")

        for name in ["new_vad_mfcc", "new_vad_logfbank"]:
            if not os.path.exists(output_dir.replace("vad_wav", name)):
                os.makedirs(output_dir.replace("vad_wav", name))

        output_path = path.replace(".wav", ".pkl")

        output_path = output_path.replace("vad_wav", "new_vad_mfcc")
        with open(output_path, "wb") as w:
            pkl.dump(np.transpose(mfcc), w)
        # output_path = output_path.replace("mfcc", "fbank")
        # with open(output_path, "wb") as w:
        #     pkl.dump(np.transpose(fbank), w)
        output_path = output_path.replace("new_vad_mfcc", "new_vad_logfbank")
        with open(output_path, "wb") as w:
            pkl.dump(np.transpose(logfbank), w)
