import os
import torch.utils.data as data
import torch
from tqdm import tqdm
from glob import iglob
import pickle as pkl
import random
import numpy as np
from sklearn import preprocessing


class ContinuousTestsetBySpeaker(data.Dataset):
    '''
    '''
    def __init__(self, utt_dir, test_pairs_dir, frames_num, window_size=100, hop=10, oneD=False, method="avgembed", normalize=False, std=False, torch=True, embedding_size=64):

        self.method = method
        self.utt_dir = utt_dir
        self.test_pairs_dir = test_pairs_dir
        self.frames_num = frames_num
        self.window_size = window_size
        self.hop = hop
        self.normalize = normalize
        self.std = std
        self.torch = torch
        self.embedding_size = embedding_size
        self.test_pairs = self.get_test_pairs()
        self.spkr2utt = self.get_spkr2utt()
        self.oneD = oneD

    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single features

        Returns:

        '''
        if index >= len(self.test_pairs):
            raise IndexError
        raw_label, gender_label, spkr_1, spot_1, spkr_2, spot_2 = self.test_pairs[index]
        feature_1 = self.spkr2utt[spkr_1][spot_1: spot_1+self.frames_num]
        feature_2 = self.spkr2utt[spkr_2][spot_2: spot_2+self.frames_num]

        label = (raw_label == '1')
        if self.method == "avgembed":
            img_1 = self.get_segment(feature_1)
            img_2 = self.get_segment(feature_2)
        else:
            img_1 = preprocessing.scale(feature_1, with_mean=self.normalize, with_std=self.std, axis=1)
            img_2 = preprocessing.scale(feature_2, with_mean=self.normalize, with_std=self.std, axis=1)

        if self.torch:
            if self.method == "avgembed":
                img_1 = torch.FloatTensor(img_1.transpose((0, 2, 1)))
                img_2 = torch.FloatTensor(img_2.transpose((0, 2, 1)))
                if not self.oneD:
                    img_1 = img_1.unsqueeze(1)
                    img_2 = img_2.unsqueeze(1)
            else:
                img_1 = torch.FloatTensor(img_1.T)
                img_2 = torch.FloatTensor(img_2.T)

        return [img_1, img_2, label, gender_label, spkr_1, spkr_2]

    def __len__(self):
        return len(self.test_pairs)

    def get_segment(self, features):

        if not self.torch:
            return features.astype(np.float64)

        network_inputs = []
        end = self.frames_num - self.window_size

        for i in range(0, self.frames_num, self.hop):
            if i > end:
                break
            frames_slice = features[i:i + self.window_size]
            frames_slice = preprocessing.scale(frames_slice, with_mean=self.normalize, with_std=self.std, axis=1)
            network_inputs.append(frames_slice)

        return np.array(network_inputs)

    def get_spkr2utt(self):
        spkr2utt = {}
        f = iglob(f"{self.utt_dir}/*.pkl")
        for path in tqdm(f, desc="getting speaker utterance..."):
            spkr = path.split('/')[-1].split('.')[0]

            with open(path, "rb") as f:
                feature = pkl.load(f)
            spkr2utt[spkr] = feature
        return spkr2utt

    def get_test_pairs(self):
        with open(f"{self.test_pairs_dir}/{self.frames_num}.pkl", "rb") as f:
            test_pairs = pkl.load(f)
        # if self.method == "eval":
        #     test_pairs = test_pairs[:10000]
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

    utt_dir = "/mnt/E/arthur.wang/aishell/aishell1/utt/logfbank/test"
    pairs_path = "/mnt/E/arthur.wang/aishell/aishell1/aishell1_test_list.txt"
    test_set = ContinuosTestsetBySpeaker(utt_dir, pairs_path)
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