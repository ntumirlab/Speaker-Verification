from Helper import Helper
import os
from glob import glob, iglob
import numpy as np
import random

if __name__ == "__main__":

    data_dir = "/mnt/E/arthur.wang/Google_speech_command" 
    f = iglob(f"{data_dir}/mfcc/*/*/*.pkl")

    speaker_dic = {}
    keyword_dic = {}
    for path in f:
        path_tokens = path.split('/')
        keyword = path_tokens[-3]
        filename = path_tokens[-1]
        speaker_id = path_tokens[-2]
        utterance_num = int(filename.split('_')[-1].split('.')[0])

        if speaker_id not in speaker_dic:
            speaker_dic[speaker_id] = {}
        if keyword not in speaker_dic[speaker_id]:
            speaker_dic[speaker_id][keyword] = []

        speaker_dic[speaker_id][keyword].append(path)

    data = []

    for key, value in keyword_dic.items():
        data.append(list(value.values()))

    with open("/mnt/E/arthur.wang/Google_speech_command/mfcc_different_keyword_test_list.txt", 'w') as mfcc, open("/mnt/E/arthur.wang/Google_speech_command/fbank_different_keyword_test_list.txt", 'w') as fbank, open("/mnt/E/arthur.wang/Google_speech_command/logfbank_different_keyword_test_list.txt", 'w') as logfbank:
        num = 22500
        for it in range(num):
            speaker = random.sample(list(speaker_dic.keys()), k=1)[0]

            while len(speaker_dic[speaker].keys()) < 2:
                speaker = random.sample(list(speaker_dic.keys()), k=1)[0]

            keywords = random.sample(list(speaker_dic[speaker].keys()), k=2)

            path_a = random.sample(speaker_dic[speaker][keywords[0]], k=1)[0]
            path_b = random.sample(speaker_dic[speaker][keywords[1]], k=1)[0]
            mfcc.write(f"1 {path_a} {path_b}\n")
            path_a = path_a.replace("mfcc", "fbank")
            path_b = path_b.replace("mfcc", "fbank")
            fbank.write(f"1 {path_a} {path_b}\n")
            path_a = path_a.replace("fbank", "logfbank")
            path_b = path_b.replace("fbank", "logfbank")
            logfbank.write(f"1 {path_a} {path_b}\n")

            speakers = random.sample(list(speaker_dic.keys()), k=2)
            while len(speaker_dic[speakers[1]]) == 1:
                speakers = random.sample(list(speaker_dic.keys()), k=2)

            keyword_a = random.sample(list(speaker_dic[speakers[0]].keys()), k=1)[0]
            path_a = random.sample(speaker_dic[speakers[0]][keyword_a], k=1)[0]

            keyword_b = random.sample(list(speaker_dic[speakers[1]].keys()), k=1)[0]


            while keyword_a == keyword_b:
                keyword_b = random.sample(list(speaker_dic[speakers[1]].keys()), k=1)[0]

            path_b = random.sample(speaker_dic[speakers[1]][keyword_b], k=1)[0]

            mfcc.write(f"0 {path_a} {path_b}\n")
            path_a = path_a.replace("mfcc", "fbank")
            path_b = path_b.replace("mfcc", "fbank")
            fbank.write(f"0 {path_a} {path_b}\n")
            path_a = path_a.replace("fbank", "logfbank")
            path_b = path_b.replace("fbank", "logfbank")
            logfbank.write(f"0 {path_a} {path_b}\n")