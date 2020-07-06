import pickle as pkl
from glob import iglob
from tqdm import tqdm
import random
import numpy as np
import os

test_types = ["test", "valid"]

feature_type = "vad_mfcc"
# meta_path = "/mnt/E/arthur.wang/aishell/aishell1/speaker.info"
# meta_path = "/mnt/E/arthur.wang/SRE10/wav/sre10_meta.tsv"
meta_path = "/mnt/E/arthur.wang/vox1/vox1_meta.csv"

spkr2gender = {}
with open(meta_path, 'r') as ff:
    data = ff.readlines()

# for line in data:
#     line = line.strip().split()
#     spkr2gender[f"S{line[0]}"] = line[1]

# data = data[1:]
# for line in data:
#     line = line.strip().split('\t')
#     spkr2gender[line[0]] = line[1]


# vox1
data = data[1:]
for line in data:
    line = line.strip().split('\t')
    spkr2gender[line[0]] = line[2]
features_list = []

for test_type in test_types:
    # test_dir = f"/mnt/E/arthur.wang/aishell/aishell1/speaker_utt/vad_mfcc/{test_type}"
    test_dir = f"/mnt/E/arthur.wang/vox1/speaker_utt/vad_mfcc/{test_type}"

    features = {}
    f = iglob(f"{test_dir}/*.pkl")
    for path in tqdm(f, desc="getting speaker utterance..."):
        spkr = path.split('/')[-1].split('.')[0]

        with open(path, "rb") as f:
            feature = pkl.load(f)
        features[spkr] = feature
    features_list.append(features)
    # if test_type != "mix":
    # continue

    times = [1, 2, 4, 8, 16]

    for time in tqdm(times):
        num_frames = time * 100
        feature_spots = {}

        for spkr, feature in features.items():
            spots = []
            for j in range(0, len(feature), num_frames):
                if j > len(feature) - num_frames:
                    break
                spots.append(j)
            random.shuffle(spots)

            feature_spots[spkr] = spots

        num = 22500
        test_pairs = []
        for i in tqdm(range(num)):

            spkr = random.sample(feature_spots.keys(), 1)[0]
            while(len(feature_spots[spkr]) < 2):
                spkr = random.sample(feature_spots.keys(), 1)[0]
            feature_1, feature_2 = random.sample(feature_spots[spkr], 2)

            test_pairs.append(('1', '1', spkr, feature_1, spkr, feature_2))

            spkr_1, spkr_2 = random.sample(feature_spots.keys(), 2)
            while(len(feature_spots[spkr_1])<1 or len(feature_spots[spkr_2])<1):
                spkr_1, spkr_2 = random.sample(feature_spots.keys(), 2)
            feature_1 = random.sample(feature_spots[spkr_1], 1)[0]
            feature_2 = random.sample(feature_spots[spkr_2], 1)[0]
            gender_1 = spkr2gender[spkr_1]
            gender_2 = spkr2gender[spkr_2]
            if gender_1 == gender_2:
                gender_label = '1'
            else:
                gender_label = '0'
            test_pairs.append(('0', gender_label, spkr_1, feature_1, spkr_2, feature_2))

        # new_dir = f"/mnt/E/arthur.wang/aishell/aishell1/test_pair/{feature_type}/{test_type}"
        new_dir = f"/mnt/E/arthur.wang/vox1/test_pair/{feature_type}/{test_type}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # with open(f"/mnt/E/arthur.wang/aishell/aishell1/test_pair/{feature_type}/{test_type}/{num_frames}.pkl", "wb") as w:
        with open(f"/mnt/E/arthur.wang/vox1/test_pair/{feature_type}/{test_type}/{num_frames}.pkl", "wb") as w:
            pkl.dump(test_pairs, w)



exit()
test_type = "mix"

test_features = features_list[0]
valid_features = features_list[1]

times = [1, 2, 4, 8, 16]

for time in tqdm(times):
    num_frames = time * 100
    test_feature_spots = {}

    for spkr, feature in test_features.items():
        spots = []
        for j in range(0, len(feature), num_frames):
            if j > len(feature) - num_frames:
                break
            spots.append(j)
        random.shuffle(spots)
        test_feature_spots[spkr] = spots

    valid_feature_spots = {}

    for spkr, feature in valid_features.items():
        spots = []
        for j in range(0, len(feature), num_frames):
            if j > len(feature) - num_frames:
                break
            spots.append(j)
        random.shuffle(spots)
        valid_feature_spots[spkr] = spots


    num = 11250
    test_pairs = []
    for i in tqdm(range(num)):
        spkr = random.sample(test_feature_spots.keys(), 1)[0]
        while(len(test_feature_spots[spkr]) < 2):
            spkr = random.sample(test_feature_spots.keys(), 1)[0]
        feature_1, feature_2 = random.sample(test_feature_spots[spkr], 2)
        test_pairs.append(('1', '1', spkr, feature_1, spkr, feature_2))

        spkr = random.sample(valid_feature_spots.keys(), 1)[0]
        while(len(valid_feature_spots[spkr]) < 2):
            spkr = random.sample(valid_feature_spots.keys(), 1)[0]
        feature_1, feature_2 = random.sample(valid_feature_spots[spkr], 2)
        test_pairs.append(('1', '1', spkr, feature_1, spkr, feature_2))

        spkr_1, spkr_2 = random.sample(test_feature_spots.keys(), 2)
        while(len(test_feature_spots[spkr_1])<1 or len(test_feature_spots[spkr_2])<1):
            spkr_1, spkr_2 = random.sample(test_feature_spots.keys(), 2)
        feature_1 = random.sample(test_feature_spots[spkr_1], 1)[0]
        feature_2 = random.sample(test_feature_spots[spkr_2], 1)[0]
        gender_1 = spkr2gender[spkr_1]
        gender_2 = spkr2gender[spkr_2]

        spkr_3, spkr_4 = random.sample(valid_feature_spots.keys(), 2)
        while(len(feature_spots[spkr_3])<1 or len(valid_feature_spots[spkr_4])<1):
            spkr_3, spkr_4 = random.sample(valid_feature_spots.keys(), 2)
        feature_3 = random.sample(valid_feature_spots[spkr_3], 1)[0]
        feature_4 = random.sample(valid_feature_spots[spkr_4], 1)[0]
        gender_3 = spkr2gender[spkr_3]
        gender_4 = spkr2gender[spkr_4]

        if gender_1 == gender_3:
            gender_label_1 = '1'
        else:
            gender_label_1 = '0'

        if gender_2 == gender_4:
            gender_label_2 = '1'
        else:
            gender_label_2 = '0'
        test_pairs.append(('0', gender_label_1, spkr_1, feature_1, spkr_3, feature_3))
        test_pairs.append(('0', gender_label_2, spkr_2, feature_2, spkr_4, feature_4))



    # new_dir = f"/mnt/E/arthur.wang/aishell/aishell1/test_pair/{feature_type}/{test_type}"
    new_dir = f"/mnt/E/arthur.wang/vox1/test_pair/{feature_type}/{test_type}"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # with open(f"/mnt/E/arthur.wang/aishell/aishell1/test_pair/{feature_type}/{test_type}/{num_frames}.pkl", "wb") as w:
    with open(f"/mnt/E/arthur.wang/vox1/test_pair/{feature_type}/{test_type}/{num_frames}.pkl", "wb") as w:
        pkl.dump(test_pairs, w)