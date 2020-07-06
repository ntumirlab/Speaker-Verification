import os
from glob import glob, iglob
import numpy as np
from sklearn import preprocessing
import random

if __name__ == "__main__":

    data_path = "/mnt/E/arthur.wang/SRE10_WAV/train/8conv.trn"
    output_path = "/mnt/E/arthur.wang/SRE10_WAV"
    save_path = f"{output_path}/sre10_meta.tsv"
    with open(data_path, 'r') as f:
        lines = f.readlines()

    test_index = random.sample(range(len(lines)), k=int(len(lines)/3))

    with open(save_path, 'w') as w:
        w.write("id\tgender\tset\tpaths\n")
        for i, line in enumerate(lines):
            line = line.strip().split(' ')
            speaker_id = line[0]
            gender = line[1]
            paths = ' '.join(line[2:])
            set = "dev"
            if i in test_index:
                set = "tets"
            w.write(f"{speaker_id}\t{gender}\t{set}\t{paths}\n")
