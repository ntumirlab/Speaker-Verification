from Helper import Helper
import os
from glob import glob, iglob
import numpy as np
import random

if __name__ == "__main__":

    data_dir = "/mnt/E/arthur.wang/Google_speech_command/" 
    f = iglob(f"{data_dir}/mfcc/*/*/*.pkl")

    keyword_dic = {}
    for path in f:
        path_tokens = path.split('/')
        keyword = path_tokens[-3]
        filename = path_tokens[-1]
        speaker_id = path_tokens[-2]
        utterance_num = int(filename.split('_')[-1].split('.')[0])

        if keyword not in keyword_dic:
            keyword_dic[keyword] = {}
        if speaker_id not in keyword_dic[keyword]:
            keyword_dic[keyword][speaker_id] = []

        keyword_dic[keyword][speaker_id].append(path)

    data = []        

    for key, value in keyword_dic.items():
        data.append(list(value.values()))

    with open("/mnt/E/arthur.wang/Google_speech_command/mfcc_same_keyword_test_list.txt", 'w') as mfcc, open("/mnt/E/arthur.wang/Google_speech_command/fbank_same_keyword_test_list.txt", 'w') as fbank, open("/mnt/E/arthur.wang/Google_speech_command/logfbank_same_keyword_test_list.txt", 'w') as logfbank:
        for speakers in data:
            num = 750
            for it in range(num):
                index = random.sample(range(len(speakers)), k=1)[0]
                while len(speakers[index]) < 2:
                    index = random.sample(range(len(speakers)), k=1)[0]
                paths = random.sample(speakers[index], k=2)
                mfcc.write(f"1 {paths[0]} {paths[1]}\n")
                path_a = paths[0].replace("mfcc", "fbank")
                path_b = paths[1].replace("mfcc", "fbank")
                fbank.write(f"1 {path_a} {path_b}\n")
                path_a = path_a.replace("fbank", "logfbank")
                path_b = path_b.replace("fbank", "logfbank")
                logfbank.write(f"1 {path_a} {path_b}\n")

                indexs = random.sample(range(len(speakers)), k=2)
                path_a = random.sample(speakers[indexs[0]], k=1)[0]
                path_b = random.sample(speakers[indexs[1]], k=1)[0]
                mfcc.write(f"0 {path_a} {path_b}\n")
                path_a = path_a.replace("mfcc", "fbank")
                path_b = path_b.replace("mfcc", "fbank")
                fbank.write(f"0 {path_a} {path_b}\n")
                path_a = path_a.replace("fbank", "logfbank")
                path_b = path_b.replace("fbank", "logfbank")
                logfbank.write(f"0 {path_a} {path_b}\n")