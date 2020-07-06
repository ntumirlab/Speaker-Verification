from tqdm import tqdm
from glob import iglob
import random

if __name__ == "__main__":
    data_dir = "/mnt/E/arthur.wang/Techine/logfbank"

    f = iglob(f"{data_dir}/*/*/*.pkl")
    speaker_dic = {}
    for path in tqdm(f, total=21505):

        path = path.replace(f"{data_dir}/", "")

        path_tokens = path.split('/')
        speaker_id = path_tokens[0]

        if speaker_id not in speaker_dic:
            speaker_dic[speaker_id] = []

        speaker_dic[speaker_id].append(path)

    with open(f"/mnt/E/arthur.wang/Techine/Techine_test_list.txt", 'w') as w:

        num = 22500
        for _ in range(num):
            # target
            speaker = random.sample(list(speaker_dic.keys()), k=1)[0]

            while len(speaker_dic[speaker]) < 2:
                speaker = random.sample(list(speaker_dic.keys()), k=1)[0]

            paths = random.sample(speaker_dic[speaker], k=2)
            w.write(f"1 {paths[0]} {paths[1]}\n")

            speakers = random.sample(list(speaker_dic.keys()), k=2)

            path_a = random.sample(speaker_dic[speakers[0]], k=1)[0]
            path_b = random.sample(speaker_dic[speakers[1]], k=1)[0]
            w.write(f"0 {path_a} {path_b}\n")
