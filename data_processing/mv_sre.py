import os
from tqdm import tqdm

with open("/mnt/E/arthur.wang/SRE10/wav/sre10_meta.tsv", 'r') as f:
    meta = [line.strip().split('\t') for line in f.readlines()]
meta = meta[1:]

for line in tqdm(meta):
    speaker_id = line[0]
    gender = line[1]
    speaker_set = line[2]

    data_dir = f"/mnt/E/arthur.wang/SRE10/"
    for name in ["sep_vad_wav", "mfcc", "fbank", "logfbank"]:
        os.system(f"mv {data_dir}/{name}/data/phonecall/tel/{speaker_id} {data_dir}/{name}/data/phonecall/tel/{speaker_set}/")
