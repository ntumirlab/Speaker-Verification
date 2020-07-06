from tqdm import tqdm
from glob import iglob
import random

if __name__ == "__main__":
    data_dir = "/mnt/E/arthur.wang/Techine/logfbank"

    f = iglob(f"{data_dir}/*/*/*.pkl")
    same = 0
    dif = 0
    utt_dic = {}
    for path in tqdm(f, total=3066):

        path = path.replace(f"{data_dir}/", "")

        path_tokens = path.split('/')
        speaker_id = path_tokens[0]
        utt_id = path_tokens[1]

        if utt_id not in utt_dic:
            utt_dic[utt_id] = {}

        if speaker_id not in utt_dic[utt_id]:
            utt_dic[utt_id][speaker_id] = []
        utt_dic[utt_id][speaker_id].append(path)

    eng = ["U0001", "U0002", "U0003", "U0004", "U0005", "U0006", "U0007", "U0008", "U0009", "U0010"]
    with open(f"/mnt/E/arthur.wang/Techine/Techine_test_list_same_utt.txt", 'w') as w:
        for utt_id in tqdm(list(utt_dic.keys())):

            if utt_id in eng:
                continue
            utt2spkr = {}

            utt_list = []
            for spkr, utts in utt_dic[utt_id].items():
                for utt in utts:
                    utt2spkr[utt] = spkr
                    utt_list.append(utt)
            utt_num = len(utt_list)

            for i in range(utt_num):
                for j in range(i + 1, utt_num):
                    utt_0 = utt_list[i]
                    utt_1 = utt_list[j]
                    if utt2spkr[utt_0] == utt2spkr[utt_1]:
                        label = '1'
                        same += 1
                    else:
                        label = '0'
                        dif += 1
                    w.write(f"{label} {utt_0} {utt_1}\n")

    print(same)
    print(dif)
