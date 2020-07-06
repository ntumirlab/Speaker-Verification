from glob import iglob
import numpy as np
import scipy.io.wavfile
import librosa
from tqdm import tqdm
import os
import pickle as pkl
import matplotlib.pyplot as plt


if __name__ == "__main__":

    logfbank_spkr_dic = {}
    frames = []

    # with open("/mnt/E/arthur.wang/Techine/speaker_info.tsv", 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip().split('\t')
    #         spkr2id[line[0]] = line[1]
    #         spkr2gender[line[0]] = line[2]
    # with open("/mnt/E/arthur.wang/Techine/utterance_info.tsv", 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip().split('\t')
    #         utt2id[line[0]] = line[1]

    data_dir = "/mnt/E/arthur.wang/Techine/wav"

    f = iglob(f"{data_dir}/*/*/*.wav")
    for path in tqdm(f, total=6030):

        tokens = path.split('/')
        spkr = tokens[-3]
        utt = tokens[-2]
        filename = tokens[-1]

        output_dir = path.replace(f"/{filename}", "")
        output_dir2 = f"{'/'.join(tokens[:-3])}/{utt}/{spkr}"
        output_path = path.replace(".wav", ".pkl")
        output_path2 = f"{output_dir2}/{filename}"
        output_path2 = output_path2.replace(".wav", ".pkl")
        sr, audio = scipy.io.wavfile.read(path)
        print(sr)
        exit()
        try:
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

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39, n_fft=n_fft, hop_length=hop_length)
            fbank = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64)
            logfbank = librosa.power_to_db(fbank)


            mfcc = np.transpose(mfcc)
            fbank = np.transpose(fbank)
            logfbank = np.transpose(logfbank)

            if spkr not in logfbank_spkr_dic:
                logfbank_spkr_dic[spkr] = logfbank
            else:
                np.append(logfbank_spkr_dic[spkr], logfbank, 0)

            for name in ["mfcc", "fbank", "logfbank"]:
                if not os.path.exists(output_dir.replace("wav", name)):
                    os.makedirs(output_dir.replace("wav", name))
            for name in ["mfcc_by_utt", "logfbank_by_utt"]:
                if not os.path.exists(output_dir2.replace("wav", name)):
                    os.makedirs(output_dir2.replace("wav", name))

            output_path = output_path.replace("wav", "mfcc")
            with open(output_path, "wb") as w:
                pkl.dump(mfcc, w)
            output_path = output_path.replace("mfcc", "fbank")
            with open(output_path, "wb") as w:
                pkl.dump(fbank, w)
            output_path = output_path.replace("fbank", "logfbank")
            with open(output_path, "wb") as w:
                pkl.dump(logfbank, w)
            output_path2 = output_path2.replace("wav", "mfcc_by_utt")

            with open(output_path2, "wb") as w:
                pkl.dump(mfcc, w)

            output_path2 = output_path2.replace("mfcc_by_utt", "logfbank_by_utt")
            with open(output_path2, "wb") as w:
                pkl.dump(logfbank, w)
        except:
            print(path)
            input()
    for spk, data in tqdm(logfbank_spkr_dic.items()):
        frames.append(len(data)*0.025)
        new_path = f"/mnt/E/arthur.wang/Techine/speaker_utt/{spk}.pkl"
        with open(new_path, "wb") as w:
            pkl.dump(data, w)
    plt.hist(frames)
    plt.xlabel('second')
    plt.ylabel('num')
    plt.savefig("Techine_utterance_by_speaker.png")