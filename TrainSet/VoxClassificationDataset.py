from __future__ import print_function
from tqdm import tqdm
import pickle as pkl
from glob import iglob
import torch.utils.data as data
import random
import torch
import numpy as np
from sklearn import preprocessing


class ClassificationDataset(data.Dataset):

    def __init__(self, utt_dir="/mnt/E/arthur.wang/vox1/mfcc/dev", iterations, batch_size=32, random_window=False, oneD=False, normalize=False, std=False, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(dir))

        self.iterations = iterations
        self.iterations_left = iterations
        self.features = self.get_features(utt_dir)
        self.n_classes = len(self.features)
        self.oneD = oneD
        self.batch_size = batch_size
        self.normalize = normalize
        self.std = std
        self.random_window = random_window

    def __getitem__(self, index):
        if self.iterations_left <= 0:
            self.iterations_left = self.iterations
            raise IndexError

        frames, spkrs = [], []
        if self.random_window:
            window_size = random.sample([100, 200, 300, 400, 500, 600, 800], 1)[0]
        else:
            window_size = 400

        batch = min(self.batch_size, self.iterations_left)
        for _ in range(batch):
            spkr = random.sample(range(self.n_classes), 1)[0]

            start_frame = random.sample(range(len(self.features[spkr]) - window_size), 1)[0]
            do_reverse = random.sample([1, -1], 1)[0]
            frame = self.features[spkr][start_frame: start_frame + window_size]
            if do_reverse == 1:
                frame = np.flip(frame, 0)
            # print(frame.shape)
            frame = preprocessing.scale(frame, with_mean=self.normalize, with_std=self.std, axis=1)
            frame = torch.FloatTensor(frame.T)
            if not self.oneD:
                frame = torch.unsqueeze(frame, 0)
            frames.append(frame)
            spkrs.append(spkr)

        frames = torch.stack(frames)
        spkrs = torch.LongTensor(spkrs)
        self.iterations_left -= batch
        return frames, spkrs

    def __len__(self):
        return self.iterations

    def get_features(self, dir):
        features = []
        f = iglob(f"{dir}/*.pkl")
        for path in tqdm(f, desc="getting speaker utterance..."):

            with open(path, "rb") as f:
                feature = pkl.load(f)
            features.append(feature)
        return features

if __name__ == "__main__":
    a = [1,1,1,1,1,1]
    b = [0,0,0,0,0,0]
    c = []
    c.append(a)
    c.append(b)

    c = torch.FloatTensor(c)
    print(c)
    print(preprocessing.scale(c, with_mean=True, with_std=True, axis=1))
    print(preprocessing.scale(c, with_mean=True, with_std=True))