from glob import iglob
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

feature_type = "new_vad_mfcc"

data_dir = f"/mnt/E/arthur.wang/aishell/aishell1/{feature_type}"
data_dir = f"/mnt/E/arthur.wang/SRE10/{feature_type}"
data_dir = f"/mnt/E/arthur.wang/vox1/{feature_type}"


f = iglob(f"{data_dir}/train/*/*/*.pkl")

c= 0

dic = {}
frames = []
i = 0
for path in tqdm(f):

    spk = path.split('/')[-3]

    with open(path, "rb") as f:
        data = pkl.load(f)
    if spk not in dic:
        dic[spk] = data
    else:
        dic[spk] = np.append(dic[spk], data, 0)
train_dir = data_dir.replace(feature_type, f"speaker_utt/{feature_type}/train")
valid_dir = data_dir.replace(feature_type, f"speaker_utt/{feature_type}/valid")
# test_dir = data_dir.replace(feature_type, f"speaker_utt/{feature_type}/test")


if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
# if not os.path.exists(test_dir):
    # os.makedirs(test_dir)
frames = []

for spk, data in tqdm(dic.items()):
    frames.append(len(data)*0.01)

    train_path = train_dir + f"/{spk}.pkl"
    with open(train_path, "wb") as w:
        pkl.dump(data[:-int(len(data)/10)], w)

    valid_path = valid_dir + f"/{spk}.pkl"
    with open(valid_path, "wb") as w:
        pkl.dump(data[int(len(data)/10)*9:], w)

    # test_path = test_dir + f"/{spk}.pkl"
    # with open(test_path, "wb") as w:
    #     pkl.dump(data, w)

# plt.hist(frames)
# plt.xlabel('second')
# plt.ylabel('num')
# plt.savefig("vox1_train_duration.png")