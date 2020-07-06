from glob import iglob
import pickle as pkl
import os
import subprocess
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



c= 0

spkr2id = {}
spkr2gender = {}
utt2id = {}
paths = []
frames = []


with open("/mnt/E/arthur.wang/Techine/speaker_info.tsv", 'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        spkr2id[line[0]] = line[1]
        spkr2gender[line[0]] = line[2]
with open("/mnt/E/arthur.wang/Techine/utterance_info.tsv", 'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        utt2id[line[0]] = line[1]

data_dir = "/mnt/E/arthur.wang/鈦映科技-2008-語者辨識"

f = iglob(f"{data_dir}/*/*/*.wav")

for path in tqdm(f, total=6030):

    tokens = path.split('/')

    pre_path = '/'.join(tokens[:-4])

    spkr = tokens[-2].split('#')[0]
    spkr = spkr2id[spkr]
    utt, utt_num = tokens[-1].split('#')
    utt = utt2id[utt]
    new_dir = f"{pre_path}/Techine/wav/{spkr}/{utt}"
    new_path = f"{new_dir}/{utt}_{utt_num}"

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    path = path.replace("\'", "\\'" )
    cmd = f"cp {path} {new_path}"

    retcode = subprocess.call(cmd, shell=True)
    if retcode:
        print(cmd)
        input()


    c+=1
