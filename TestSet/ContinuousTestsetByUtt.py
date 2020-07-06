import os
import torch.utils.data as data
import torch
from tqdm import tqdm
from glob import iglob
import pickle as pkl
import random
import numpy as np
from sklearn import preprocessing


class ContinuousTestsetByUtt(data.Dataset):
    '''
    '''
    def __init__(self, feature_dir, pairs_path, spkr2gender, hop=10, window_size=32, normalize=False, std=False, torch=True, embedding_size=64):

        self.feature_dir = feature_dir
        self.pairs_path = pairs_path
        self.spkr2gender = spkr2gender
        self.test_pairs = self.get_test_pair(self.pairs_path)
        self.hop = hop
        self.window_size = window_size
        self.normalize = normalize
        self.std = std
        self.torch = torch
        self.embedding_size = embedding_size

    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single features

        Returns:

        '''

        path_1, path_2, label, gender_label, spkr_1, spkr_2 = self.test_pairs[index]

        try:
            with open(path_1, "rb") as f:
                utt_1 = pkl.load(f)
        except:
            with open(path_1.replace("vad_mfcc", "mfcc"), "rb") as f:
                utt_1 = pkl.load(f)
        try:
            with open(path_2, "rb") as f:
                utt_2 = pkl.load(f)
        except:
            with open(path_2.replace("vad_mfcc", "mfcc"), "rb") as f:
                utt_2 = pkl.load(f)

        if self.torch:
            img_1 = torch.FloatTensor(preprocessing.scale(utt_1, with_mean=self.normalize, with_std=self.std, axis=1).T)
            img_2 = torch.FloatTensor(preprocessing.scale(utt_2, with_mean=self.normalize, with_std=self.std, axis=1).T)


        # img_1 = self.get_segment(utt_1)
        # img_2 = self.get_segment(utt_2)
        # if self.torch:
        #     img_1 = torch.FloatTensor(img_1.transpose((0, 2, 1)))
        #     img_2 = torch.FloatTensor(img_2.transpose((0, 2, 1)))

        return [img_1, img_2, label, gender_label, spkr_1, spkr_2]

    def __len__(self):
        return len(self.test_pairs)

    def get_segment(self, features):
        network_inputs = []
        total_frames = len(features)

        features = preprocessing.scale(features, with_mean=self.normalize, with_std=self.std)
        if not self.torch:
            return features.astype(np.float64)
        end = total_frames - self.window_size

        for i in range(0, total_frames, self.hop):
            if i > end:
                break
            frames_slice = features[i:i + self.window_size]
            network_inputs.append(frames_slice)

        # for _ in range(self.augment):
        #     network_inputs += network_inputs

        return np.array(network_inputs)


    def get_test_pair(self, pairs_path):
        data = [line.strip().split(' ') for line in open(pairs_path, 'r').readlines()]

        test_pairs = []

        for line in tqdm(data):
            if line[0] == '1':
                label = True
            else:
                label = False

            path_1 = f"{self.feature_dir}/{line[1]}"
            path_2 = f"{self.feature_dir}/{line[2]}"
            spkr_1 = line[1].split('/')[0]
            spkr_2 = line[2].split('/')[0]
            gender_1 = self.spkr2gender[spkr_1]
            gender_2 = self.spkr2gender[spkr_2]
            if gender_1 == gender_2:
                gender_label = '1'
            else:
                gender_label = '0'
            test_pairs.append((path_1, path_2, label, gender_label, spkr_1, spkr_2))

        return test_pairs

    @staticmethod
    def custom_collate_fn(batch):

        segment_length_1 = []
        segment_length_2 = []
        label = []
        gender = []
        for pair in batch:
            segment_length_1.append(len(pair[0]))
            segment_length_2.append(len(pair[1]))
            label.append(pair[2])
            gender.append(pair[3])
        img_1, img_2, _, _ = zip(*batch)

        return torch.cat(img_1), torch.cat(img_2), label, gender, segment_length_1, segment_length_2

if __name__ == "__main__":

    feature_dir = "/mnt/E/arthur.wang/aishell/aishell1/utt/logfbank/test"
    pairs_path = "/mnt/E/arthur.wang/aishell/aishell1/aishell1_test_list.txt"
    test_set = ContinuosTestsetBySpeaker(feature_dir, pairs_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, collate_fn=test_set.custom_collate_fn)
    for data_a, data_b, label, length_a, length_b in test_loader:
        print(type(data_a))
        exit()
        data_a = torch.unsqueeze(data_a, 1)
        start_a = 0
        start_b = 0
        embedding_a = []
        for segment_a, segment_b in zip(length_a, length_b):
            temp = data_a[start_a:segment_a]
            temp = torch.mean(temp, 0)
            embedding_a.append(temp)
        print(len(embedding_a))
        embedding_a = torch.cat(embedding_a, 0)
        print(embedding_a.shape)
        exit()
    # feature_1, feature_2, label = test_set[0]
    # print(feature_1.shape)