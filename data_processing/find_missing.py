from glob import iglob
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
def read_PKL(filename, normalize=True, std=False):
    #audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()
    # filename = filename.replace(".wav", ".pkl")
    with open(filename, "rb") as f:
        if normalize:
            if std:
                audio = preprocessing.scale(pkl.load(f))
            else:
                audio = preprocessing.scale(pkl.load(f), with_std=False)
        else:
            audio = pkl.load(f)
    return audio
# data_dir = "/mnt/E/arthur.wang/aishell/aishell1/opsm"

# f = iglob(f"{data_dir}/*/*/*.pkl")

data_dir = "/mnt/E/arthur.wang/vox1/logfbank"
f = iglob(f"{data_dir}/*/*/*/*.pkl")

error_list = []

c= 0
# dic = {}
# for i in range(80001, 80446):
#     dic[str(i)] = []

for path in tqdm(f):


    # data = read_PKL(path)
    # new_path = path.replace("logfbank", "normalize_logfbank")
    # new_dir = '/'.join(new_path.split('/')[:-1])

    # if not os.path.exists(new_dir):
    #     os.makedirs(new_dir)
    # # with open(path, "rb") as ff:
        # data = pkl.load(ff)
    # with open(new_path, 'wb') as ww:
    #     pkl.dump(data, ww)
    # # if i > 78989:
    #     try:
    #         with open(path, "rb") as ff:
    #             data = pkl.load(ff)
    #     except:
    #         error_list.append(path)
    #         continue

    #     if len(data)==0:
    #         error_list.append(path)
    # i+=1
    # if(len(data)==0):
    #     os.system(f"rm {path}")
        # c-=1
    # with open(path, "rb") as f:
    #     data = pkl.load(f)
    # if len(data) < 40:
        # print(len(data), path)
    # os.system(f"rm {path}")
    c+=1
        # os.system(f"rm {path}")
    # os.system(f"rm {path}")
#     speaker_id = path.split('/')[-1].split('.')[0].split('_')[-1]
#     dic[speaker_id].append(path)
print(c)
# print(len(error_list))
# input()
# for line in tqdm(error_list):
#     os.system(f"rm {line}")
# # for speaker_id, paths in dic.items():
# #     if len(paths) != 8:
# #         print(speaker_id)
# #         for path in paths:
# #             print(path)