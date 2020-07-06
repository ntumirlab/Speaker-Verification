from glob import iglob
import pickle as pkl
import os
import random

data_dir = "/mnt/E/arthur.wang/vox1/wav"

f = iglob(f"{data_dir}/*/*/*/*.wav")

c= 0
speaker_dic = {}
for path in f:
    speaker_id = path.split('/')[-3]

    if speaker_id not in speaker_dic:
        speaker_dic[speaker_id] = []

    speaker_dic[speaker_id].append(path.replace("/mnt/E/arthur.wang/vox1/wav/", ''))


with open("/mnt/E/arthur.wang/vox1/enroll_list", 'w') as enr, open("/mnt/E/arthur.wang/vox1/test_list_for_enroll.txt", 'w') as tes:
    for speaker, paths in speaker_dic.items():
        enrollments = random.sample(range(len(paths)), 3)
        tests = enrollments[5:]
        enrollments = enrollments[:5]
        for i in enrollments:
            enr.write(f"{speaker} {paths[i]}\n")
        for i in tests:
            tes.write(f"{speaker} {paths[i]}\n")
