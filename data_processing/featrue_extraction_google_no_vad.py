from glob import iglob
import numpy as np
import scipy.io.wavfile
import librosa
from tqdm import tqdm
import os
import pickle as pkl
# from audio_processing import mk_MFB
from python_speech_features import fbank, delta


if __name__ == "__main__":

    data_dir = "/mnt/E/arthur.wang/Google_speech_command"

    f = iglob(f"{data_dir}/no_vad_wav/*/*/*.wav")
    # f = iglob(f"{data_dir}fbank/*/*/*.npy")
    c=0
    for path in tqdm(f, total=64721):
        c+=1

        sr, audio = scipy.io.wavfile.read(path)

        if audio.dtype == 'int16':
            nb_bits = 16  # -> 16-bit wav files
        elif audio.dtype == 'int32':
            nb_bits = 32  # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        audio = audio / (max_nb_bit + 1.0)

        audio = librosa.effects.preemphasis(audio)

        hop_duration = 0.01  # in seconds
        win_duration = 0.025
        hop_length = int(sr * hop_duration)
        n_fft = int(sr * win_duration)
        try:
            # mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39, n_fft=n_fft, hop_length=hop_length)
            # fbank = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64)
            # logfbank = librosa.power_to_db(fbank)
            filter_banks, energies = fbank(audio, samplerate=sr, nfilt=64, winlen=0.025)
        except:
            print(path)
            input()
        output_dir = '/'.join(path.split('/')[:-1])

        for name in ["no_vad_mfcc", "no_vad_fbank", "no_vad_logfbank", "no_vad_psf_fbank"]:
            if not os.path.exists(output_dir.replace("no_vad_wav", name)):
                os.makedirs(output_dir.replace("no_vad_wav", name))

        output_path = path.replace(".wav", ".pkl")

        output_path = output_path.replace("no_vad_wav", "no_vad_psf_fbank")
        with open(output_path, "wb") as w:
            pkl.dump(filter_banks, w)
        # output_path = output_path.replace("no_vad_wav", "no_vad_mfcc")
        # with open(output_path, "wb") as w:
        #     pkl.dump(np.transpose(mfcc), w)
        # output_path = output_path.replace("no_vad_mfcc", "no_vad_fbank")
        # with open(output_path, "wb") as w:
        #     pkl.dump(np.transpose(fbank), w)
        # output_path = output_path.replace("no_vad_fbank", "no_vad_logfbank")
        # with open(output_path, "wb") as w:
        #     pkl.dump(np.transpose(logfbank), w)
    print(c)