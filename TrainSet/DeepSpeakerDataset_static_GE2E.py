from __future__ import print_function

from tqdm import tqdm
import numpy as np

import torch.utils.data as data


def find_classes(voxceleb):
    classes = list(set([datum['speaker_id'] for datum in voxceleb]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def generate_triplets(features, num_triplets, n_classes, transform):
    def create_indices(_features):
        inds = dict()
        genders_inds = {}
        for _, (feature_path, speaker_id, gender) in enumerate(_features):
            if speaker_id not in inds:
                inds[speaker_id] = []
            inds[speaker_id].append(feature_path)
            genders_inds[speaker_id] = gender
        return inds, genders_inds

    triplets = []
    # Indices = array of labels and each label is an array of indices
    indices, gender_indices = create_indices(features)
    pbar = tqdm(total=num_triplets)
    triplet_count = 0
    # for x in tqdm(range(num_triplets)):
    while triplet_count < num_triplets:
        try:
            c1 = np.random.randint(0, n_classes)
            c2 = np.random.randint(0, n_classes)
            while len(indices[c1]) < 2:
                c1 = np.random.randint(0, n_classes)

            # while c1 == c2 or not gender_indices[c1] == gender_indices[c2]:
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            if len(indices[c2]) ==1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices[c2]) - 1)

            triplets.append([transform(indices[c1][n1]), transform(indices[c1][n2]), transform(indices[c2][n3]), c1, c2])
            pbar.update(1)
            triplet_count += 1
        except:
            continue
    return triplets



class DeepSpeakerDataset(data.Dataset):

    def __init__(self, voxceleb, dir, n_triplets,loader, transform=None, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(dir))
        #voxceleb = read_voxceleb_structure(dir)

        #voxceleb = voxceleb[voxceleb['subset'] == 'dev']

        #voxceleb = voxceleb[1:5000]
        #voxceleb = voxceleb[445:448]
        self.loader = loader

        if len(voxceleb) == 0:
            raise(RuntimeError(('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')))

        classes, class_to_idx = find_classes(voxceleb)

        features = []
        female_features = []
        male_features = []
        cc = 0
        for vox_item in tqdm(voxceleb):
            try:
                item = (self.loader(vox_item['file_path']), class_to_idx[vox_item['speaker_id']], vox_item["gender"])
                features.append(item)
            except:
                cc+=1
                continue
        print(f"skipped {cc}")

        self.root = dir
        self.features = features
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.n_triplets = n_triplets

        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = generate_triplets(self.features, self.n_triplets,len(self.classes), self.transform)



    def __getitem__(self, index):
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

        # Get the index of each features in the triplet
        # a, p, n, c1, c2 = self.training_triplets[index]

        # transform features if required
        # feature_a, feature_p, feature_n = self.transform(a), self.transform(p), self.transform(n)

        feature_a, feature_p, feature_n, c1, c2 = self.training_triplets[index]
        return feature_a, feature_p, feature_n, c1, c2

    def __len__(self):
        return len(self.training_triplets)