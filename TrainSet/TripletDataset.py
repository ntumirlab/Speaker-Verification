from __future__ import print_function
from tqdm import tqdm
import pickle as pkl
from glob import iglob
import torch.utils.data as data
import random
import torch
import numpy as np
from sklearn import preprocessing


class TripletDataset(data.Dataset):

    def __init__(self, utt_dir, n_triplets, batch_size=32, window_size=100, oneD=False, random_window=False, normalize=False, std=False, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(dir))

        self.n_triplets = n_triplets
        self.triplets_left = n_triplets
        self.batch_size = batch_size
        self.features = self.get_features(utt_dir)
        self.n_classes = len(self.features)
        self.window_size = window_size
        self.oneD = oneD
        self.random_window = random_window
        self.normalize = normalize
        self.std = std

    def __getitem__(self, index):
        # print("get_item")
        '''

        Args:
            index: Index of the triplet or the matches - not of a single feature

        Returns:

        '''
        # def transform(feature_path):
        #     """Convert image into numpy array and apply transformation
        #        Doing this so that it is consistent with all other datasets
        #     """

        #     feature = self.loader(feature_path)
        #     return self.transform(feature)

        # Get the index of each feature in the triplet
        if self.triplets_left <= 0:
            self.triplets_left = self.n_triplets
            raise IndexError
        # self.window_size = random.sample(self.window_sizes, 1)[0] * 100
        if self.random_window:
            window_size = random.sample([100, 200, 300, 400, 500, 600, 800], 1)[0]
        else:
            window_size = self.window_size
        anchor, postive, negative, class1, class2 = [], [], [], [], []
        batch = min(self.batch_size, self.triplets_left)
        for _ in range(batch):
            a, p, n, c1, c2 = self.generate_triplets_call(window_size)
            anchor.append(a)
            postive.append(p)
            negative.append(n)
            class1.append(c1)
            class2.append(c2)

        anchor = torch.stack(anchor)
        postive = torch.stack(postive)
        negative = torch.stack(negative)
        class1 = torch.LongTensor(class1)
        class2 = torch.LongTensor(class2)
        self.triplets_left -= batch
        return anchor, postive, negative, class1, class2

    def __len__(self):
        return self.n_triplets

    def transform(self, img):
        img = torch.FloatTensor(img.transpose((0, 2, 1)))
        return img

    def get_features(self, dir):
        features = []
        f = iglob(f"{dir}/*.pkl")
        for path in tqdm(f, desc="Triplet: getting speaker utterance..."):

            with open(path, "rb") as f:
                feature = pkl.load(f)
            features.append(feature)
        return features

    def generate_triplets_call(self, window_size):
        # print("generate_triplets_call")
        # Indices = array of labels and each label is an array of indices
        # indices = create_indices(features)
        spkr_1, spkr_2 = random.sample(range(self.n_classes), 2)
        frame_a, frame_p = random.sample(range(len(self.features[spkr_1]) - window_size), 2)
        while abs(frame_a - frame_p) <= window_size:
            frame_a, frame_p = random.sample(range(len(self.features[spkr_1]) - window_size), 2)

        frame_n = random.sample(range(len(self.features[spkr_2]) - window_size), 1)[0]

        frame_a = self.features[spkr_1][frame_a: frame_a + window_size]
        frame_p = self.features[spkr_1][frame_p: frame_p + window_size]
        frame_n = self.features[spkr_2][frame_n: frame_n + window_size]

        frame_a = preprocessing.scale(frame_a, with_mean=self.normalize, with_std=self.std, axis=1)
        frame_p = preprocessing.scale(frame_p, with_mean=self.normalize, with_std=self.std, axis=1)
        frame_n = preprocessing.scale(frame_n, with_mean=self.normalize, with_std=self.std, axis=1)

        # do_reverse = random.sample([1, 1, 1, -1, -1, -1], 3)
        # if do_reverse[0] == 1:
        #     frame_a = np.flip(frame_a, 0)
        # if do_reverse[1] == 1:
        #     frame_p = np.flip(frame_p, 0)
        # if do_reverse[2] == 1:
        #     frame_n = np.flip(frame_n, 0)

        a = torch.FloatTensor(frame_a.T)
        p = torch.FloatTensor(frame_p.T)
        n = torch.FloatTensor(frame_n.T)
        if not self.oneD:
            a = torch.unsqueeze(a, 0)
            p = torch.unsqueeze(p, 0)
            n = torch.unsqueeze(n, 0)

        return a, p, n, spkr_1, spkr_2

if __name__ == "__main__":
    td = TripletDataset(utt_dir="/mnt/E/arthur.wang/aishell/aishell1/speaker_utt/logfbank/train", n_triplets=100)
    for a, p, n, c1, c2 in td:
        print(a.shape)
        print(p.shape)
        print(n.shape)
        print(len(c1))
        print(len(c2))
        input()
