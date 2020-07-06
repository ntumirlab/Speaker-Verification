from glob import iglob
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data_dir = "/mnt/E/arthur.wang/鈦映科技-2008-語者辨識"

f = iglob(f"{data_dir}/*/*/*.wav")

c= 0

spkr_set = set()
utt_dic = {}
spkr2id = {}
spkr2gender = {}
spkr2set = {}
utt2id = {}
paths = []
frames = []
for path in tqdm(f, total=6030):

    paths.append(path)
    tokens = path.split('/')

    if tokens[-3] == "第一次錄音檔":
        _set = "set_1"
    elif tokens[-3] == "第二次錄音檔":
        _set = "set_2"
    else:
        print("error, set not valid")
        exit()

    spkr, gender = tokens[-2].split('#')
    if gender == '1':
        gender = 'm'
    else:
        gender = 'f'
    spkr2gender[spkr] = gender
    spkr_set.add(spkr)
    utt = tokens[-1].split('#')[0]

    if utt not in utt_dic:
        utt_dic[utt] = 0
    utt_dic[utt] += 1
    spkr2set[spkr] = _set

    c+=1

spkr_list = sorted(list(spkr_set))
utt_list = sorted(list(utt_dic.keys()))


with open("/mnt/E/arthur.wang/Techine/speaker_info.tsv", 'w') as w:
    w.write("name\tid\tgender\n")
    for i, spkr in enumerate(spkr_list):
        w.write(f"{spkr}\tS{i+1:03}\t{spkr2gender[spkr]}\n")

with open("/mnt/E/arthur.wang/Techine/utterance_info.tsv", 'w') as w:
    w.write("utterance\tid\tcount\n")
    for i, utt in enumerate(utt_list):
        w.write(f"{utt}\tU{i+1:04}\t{utt_dic[utt]}\n")
