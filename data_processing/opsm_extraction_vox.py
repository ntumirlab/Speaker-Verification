from glob import iglob
import numpy as np
import scipy.io.wavfile
import librosa
from tqdm import tqdm
import os
import pickle as pkl


if __name__ == "__main__":

    data_dir = "/mnt/E/arthur.wang/vox1/wav"
    conf_path = "/home/arthur.wang/opsm/opensmile-2.3.0/config/my_IS12_speaker_trait.conf"

    f = iglob(f"{data_dir}/*/*/*/*.wav")
    c = 0
    for path in tqdm(f, total=153516):
        c+=1

        if c < 67011:
            continue
        tokens = path.split('/')
        filename = tokens[-1]

        try:
            os.system(f"SMILExtract -C {conf_path} -I {path} -D vox1Out.csv -noconsoleoutput")
            with open("./vox1Out.csv", 'r') as f:
                opsm = f.readlines()
            opsm = opsm[1:]
            for i, line in enumerate(opsm):
                opsm[i] = [np.float64(f) for f in line.split(';')[1:]]
        except:
            continue

        output_dir = path.replace(f"/{filename}", "")

        if not os.path.exists(output_dir.replace("wav", "opsm")):
            os.makedirs(output_dir.replace("wav", "opsm"))

        output_path = path.replace(".wav", ".pkl")

        output_path = output_path.replace("wav", "opsm")
        with open(output_path, "wb") as w:
            pkl.dump(np.transpose(opsm), w)
