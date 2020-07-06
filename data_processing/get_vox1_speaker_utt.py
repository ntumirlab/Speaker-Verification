from glob import iglob
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data_dir = "/mnt/E/arthur.wang/vox1/logfbank/train"

f = iglob(f"{data_dir}/*/*/*.pkl")

c= 0

dic = {}
frames = []
for path in tqdm(f):

    spk = path.split('/')[-3]

    with open(path, "rb") as f:
        data = pkl.load(f)

    if spk not in dic:
        dic[spk] = data
    else:
        dic[spk] = np.append(dic[spk], data, 0)

new_dir = data_dir.replace("logfbank/train", "speaker_utt/logfbank")
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
frames = []

for spk, data in tqdm(dic.items()):
    frames.append(len(data)*0.025)
    new_path = new_dir + f"/{spk}.pkl"
    with open(new_path, "wb") as w:
        pkl.dump(data, w)


# plt.hist(frames)
# plt.xlabel('second')
# plt.ylabel('num')
# plt.savefig("vox1_utterance_by_speaker.png")