import os
import pandas as pd
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')
wav_path = "/mnt/E/arthur.wang/aishell/aishell1/wav/dev/S0732/BAC009S0732W0397.wav"
conf_path = r"/home/arthur.wang/opsm/opensmile-2.3.0/config/my_IS12_speaker_trait_2.conf"
# conf_path = "/home/arthur.wang/opsm/opensmile-2.3.0/config/myconf.conf"
# conf_path = "/home/arthur.wang/opsm/opensmile-2.3.0/config/demo/demo1_energy.conf"
os.system(f"SMILExtract -C {conf_path} -I {wav_path} -D out.csv ")


# with open('./out.csv', 'r') as f:
#     data = f.readlines()
# print(len(data[0].split(';')))

# with open("/mnt/E/arthur.wang/vox1/logfbank/dev/id10001/1zcIwhmdeo4/00003.pkl", "rb") as f:
#     data = pkl.load(f)
#     print(len(data))