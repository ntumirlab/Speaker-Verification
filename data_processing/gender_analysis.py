import pickle as pkl
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from Helper import Helper
import bob.measure
import numpy as np
from tqdm import tqdm

def split_gender(scores, labels):
    same_scores = []
    different_scores = []
    same_labels = []
    different_labels = []
    meta_path = f"/mnt/E/arthur.wang/vox1/vox1_meta.csv"

    speaker_dic = {}

    with open(meta_path, 'r') as f:
        meta = f.readlines()
        meta = meta[1:]

    with open(test_list_path, 'r') as f:
        test_list = f.readlines()

    for line in tqdm(meta, desc="reading meta..."):
        # print(line)
        line = line.strip().split()
        speaker_dic[line[0]] = {}
        speaker_dic[line[0]]["gender"] = line[2]
        speaker_dic[line[0]]["nationality"] = line[3]

    result = []


    for i in tqdm(range(len(test_list)), desc="reading and sort..."):
        tokens = test_list[i].strip().split()

        speaker_a = tokens[1].split('/')[0]
        gender_a = speaker_dic[speaker_a]["gender"]
        nationality_a = speaker_dic[speaker_a]["nationality"]
        speaker_b = tokens[2].split('/')[0]
        gender_b = speaker_dic[speaker_b]["gender"]
        nationality_b = speaker_dic[speaker_b]["nationality"]

        if gender_a == gender_b:
            same_scores.append(scores[i])
            same_labels.append(labels[i])
        else:
            different_scores.append(scores[i])
            different_labels.append(labels[i])

    return np.array(same_scores), np.array(different_scores), np.array(same_labels), np.array(different_labels)


def score_analysis(test_list_path, scores):

    meta_path = f"/mnt/E/arthur.wang/vox1/vox1_meta.csv"

    speaker_dic = {}

    with open(meta_path, 'r') as f:
        meta = f.readlines()
        meta = meta[1:]

    with open(test_list_path, 'r') as f:
        test_list = f.readlines()

    for line in tqdm(meta, desc="reading meta..."):
        # print(line)
        line = line.strip().split()
        speaker_dic[line[0]] = {}
        speaker_dic[line[0]]["gender"] = line[2]
        speaker_dic[line[0]]["nationality"] = line[3]

    result = []

    for i in tqdm(range(len(test_list)), desc="reading and sort..."):
        tokens = test_list[i].strip().split()
        label = tokens[0] == '1'
        speaker_a = tokens[1].split('/')[0]
        gender_a = speaker_dic[speaker_a]["gender"]
        nationality_a = speaker_dic[speaker_a]["nationality"]
        speaker_b = tokens[2].split('/')[0]
        gender_b = speaker_dic[speaker_b]["gender"]
        nationality_b = speaker_dic[speaker_b]["nationality"]
        score = scores[i]
        result.append((score, label, f"{speaker_a}\t{speaker_b}\t{gender_a}\t{gender_b}\t{nationality_a}\t{nationality_b}\t"))

    sort_result = sorted(result, key=lambda x: x[0])
    total_num = len(sort_result)
    with open(f"{metrics_path}/score_list.txt", 'w') as w:
        for i in tqdm(range(total_num), desc="writing..."):
            w.write(f"{sort_result[i][0]}\t{sort_result[i][1]}\t{sort_result[i][2]}\n")

    return

def output_eer(scores, labels, context, metrics_path):

    thr, fa, fr, Cprim, Cthr = Helper.cal_Cprimary(scores, labels)

    p_scores, n_scores = scores[np.where(labels == True)].astype(np.double), scores[np.where(labels == False)[0]].astype(np.double)


    Helper.generate_det_curve(p_scores, n_scores, f'{metrics_path}/pic/det_curve/{context}_det_curve.png')
    Helper.plot_roc(labels=labels, scores=scores, p_scores=p_scores, n_scores=n_scores, output_path=f"{metrics_path}/pic/roc/{context}_roc_curve.png")
    Helper.plot_probabilty_density(p_scores=p_scores, n_scores=n_scores, output_path=f"{metrics_path}/pic/PB/{context}_prob_den.png", threshold=thr, fa=fa, fr=fr, Cprim=Cprim, Cthr=Cthr)
    Helper.plot_precision_recall(labels=labels, scores=scores, output_path=f"{metrics_path}/pic/PR/{context}_PR_curve.png")


metrics_path = "/mnt/E/arthur.wang/metrics/deepspeaker/vox1_same_gender_50softmax_training/8_normalize_logfbank"
# metrics_path = "/mnt/E/arthur.wang/metrics/deepspeaker/sre10/8_normalize_fbank"
test_input_per_file = 8
test_list_path = "/mnt/E/arthur.wang/vox1/voxceleb1_test.txt"
for epoch in range(50, 71):

    context = f"epoch{epoch}_{test_input_per_file}inputs"


    with open (f"{metrics_path}/score/{context}_scores.pkl", "rb") as f:
        scores = pkl.load(f)

    with open (f"{metrics_path}/label/{context}_labels.pkl", "rb") as f:
        labels = pkl.load(f)
    same_scores, different_scores, same_labels, different_labels = split_gender(scores, labels)

    # print(type(same_scores))
    # print(type(different_scores))
    # print(type(same_labels))
    # print(type(different_labels))
    # exit()
    output_eer(same_scores, same_labels, f"same_gender_{context}", metrics_path)
    # output_eer(different_scores, different_labels, f"diffeent_gender_{context}", metrics_path)

    # for i, score in enumerate(scores):
    #     scores[i] = score * (-1)

    # Output EER to file, and plot DET curve if required.

