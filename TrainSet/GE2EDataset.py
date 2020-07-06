import os
import torch.utils.data as data
import torch
from tqdm import tqdm
from glob import iglob
import pickle as pkl
import random
import numpy as np
from sklearn import preprocessing


class GE2EDataset(data.Dataset):
    '''
    '''
    def __init__(self, utt_dir, step=3000, speaker_num=64, utterance_num=10, window_size=100, random_window=False, oneD=True):

        self.step = step
        self.speaker_num = speaker_num
        self.utterance_num = utterance_num
        self.utt_list = self.get_utt_list(utt_dir)
        self.window_size = window_size
        self.n_classes = len(self.utt_list)
        self.random_window = random_window
        self.oneD = oneD

    def __getitem__(self, index):

        if index > self.step:
            raise IndexError

        if self.random_window:
            # window_size = 800
            # window_size, speaker_num = random.sample([(100, 200), (200, 80), (300, 60), (400, 60), (500, 40), (600, 30), (800, 30)], 1)[0]
            window_size, speaker_num = random.sample([(100, 200), (200, 80), (300, 60), (400, 60)], 1)[0]
            # window_size = random.sample([100, 200, 300, 400], 1)[0]
        else:
            window_size = self.window_size
            speaker_num = self.speaker_num
        speakers = random.sample(range(len(self.utt_list)), k=speaker_num)
        img = []
        label = []
        for spkr in speakers:
            img.append(self.get_segment(self.utt_list[spkr], window_size))
            label += [spkr] * self.utterance_num
        img = torch.stack(img)
        label = torch.LongTensor(label)
        return img, label
        # return self.features[index]

    def __len__(self):
        return self.step

    def get_segment(self, features, window_size):
        network_inputs = []
        total_frames = len(features)

        for _ in range(self.utterance_num):
            start = random.randrange(0, total_frames - window_size)

            frames_slice = features[start:start + window_size]
            # frames_slice = preprocessing.scale(frames_slice, with_mean=self.normalize, with_std=self.std)
            network_inputs.append(frames_slice)

        network_inputs = np.array(network_inputs)
        network_inputs = network_inputs.transpose((0, 2, 1))
        network_inputs = torch.FloatTensor(network_inputs)
        if not self.oneD:
            network_inputs = torch.unsqueeze(network_inputs, 1)

        return network_inputs

    def get_utt_list(self, dir):
        utt_list = []
        f = iglob(f"{dir}/*.pkl")
        for path in tqdm(f, desc="GE2E: getting speaker utterance..."):
            with open(path, "rb") as f:
                feature = pkl.load(f)
            utt_list.append(feature)

        return utt_list


# class GE2ELoss(nn.Module):
#     def __init__(self):
#         super(GE2ELoss, self).__init__()
#         self.w = nn.Parameter(th.tensor(10.0))
#         self.b = nn.Parameter(th.tensor(-5.0))

#     def forward(self, e, N, M):
#         """
#         e: N x M x D, after L2 norm
#         N: number of spks
#         M: number of utts
#         """
#         # N x D
#         c = th.mean(e, dim=1)
#         s = th.sum(e, dim=1)
#         # NM * D
#         e = e.view(N * M, -1)
#         # compute similarity matrix: NM * N
#         sim = th.mm(e, th.transpose(c, 0, 1))
#         # fix similarity matrix: eq (8), (9)
#         for j in range(N):
#             for i in range(M):
#                 cj = (s[j] - e[j * M + i]) / (M - 1)
#                 sim[j * M + i][j] = th.dot(cj, e[j * M + i])
#         # eq (5)
#         sim = self.w * sim + self.b
#         # build label N*M
#         ref = th.zeros(N * M, dtype=th.int64, device=e.device)
#         for r, s in enumerate(range(0, N * M, M)):
#             ref[s:s + M] = r
#         # ce loss
#         loss = F.cross_entropy(sim, ref)
#         return loss

if __name__ == "__main__":
    # from model import DeepSpeakerModel
    # from ge2e import GE2ELoss

    # criterion = GE2ELoss()

    # test_input = torch.rand(64, 10, 64)
    # # print(test_input[0][0])
    # # test_input = test_input.resize_(640, 1, 64, 32)
    # # print(test_input[0])
    # loss = criterion(test_input)
    # print(loss)
    # exit()

    # model = DeepSpeakerModel(embedding_size=64, num_classes=340)
    # model.to("cuda:0")
    # model.train()
    # output = model(test_input)
    # new_output = output.view(64, 10, 64)

    # for i in range(64):
    #     for j in range(10):
    #         print(i*10+j,output[i*10+j])
    #         print(i,j,new_output[i][j])
    #         input()

    utt_dir = "/mnt/E/arthur.wang/aishell/aishell1/speaker_utt/logfbank"
    # pairs_path = "/mnt/E/arthur.wang/aishell/aishell1/aishell1_test_list.txt"
    train_set = GE2EDataset(utt_dir)
    # criterion = GE2ELoss()
    for img in tqdm(train_set):
        print(img)
        # img = img.resize_(640, 1, 64, 32)
        # output = model(img)
        # output = output.view(64, 10, 64)
        # loss = GE2ELoss(output)
        # print(loss)
        # exit()

    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False, collate_fn=test_set.custom_collate_fn)
    # for data_a, data_b, label, length_a, length_b in test_loader:
    #     print(type(data_a))
    #     exit()
    #     data_a = torch.unsqueeze(data_a, 1)
    #     start_a = 0
    #     start_b = 0
    #     embedding_a = []
    #     for segment_a, segment_b in zip(length_a, length_b):
    #         temp = data_a[start_a:segment_a]
    #         temp = torch.mean(temp, 0)
    #         embedding_a.append(temp)
    #     print(len(embedding_a))
    #     embedding_a = torch.cat(embedding_a, 0)
    #     print(embedding_a.shape)
    #     exit()
    # feature_1, feature_2, label = test_set[0]
    # print(feature_1.shape)