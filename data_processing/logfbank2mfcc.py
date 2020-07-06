from glob import iglob
from tqdm import tqdm
import scipy.fftpack
import pickle as pkl

test_types = ["valid", "test", "train"]
dct_type = 2
norm = 'ortho'

utt_dir = "/mnt/E/arthur.wang/vox1/speaker_utt/logfbank"
output_dir = "/mnt/E/arthur.wang/vox1/speaker_utt/mfcc"
for test_type in test_types:
    spkr2utt = {}
    f = iglob(f"{utt_dir}/{test_type}/*.pkl")
    for path in tqdm(f, desc="getting speaker utterance..."):
        spkr = path.split('/')[-1].split('.')[0]

        with open(path, "rb") as f:
            feature = pkl.load(f)
        spkr2utt[spkr] = feature
    for spkr, utt in spkr2utt.items():
        mfcc = scipy.fftpack.dct(utt.T, axis=0, type=dct_type, norm=norm)[:39]

        output_path = f"{output_dir}/{test_type}/{spkr}.pkl"

        with open(output_path, "wb") as w:
            pkl.dump(mfcc.T, w)