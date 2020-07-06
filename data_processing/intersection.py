import pickle as pkl
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from Helper import Helper
import bob.measure
import numpy as np
metrics_path = "/mnt/E/arthur.wang/metrics/deepspeaker/vox1/8_normalize_logfbank"
for epoch in range(50, 51):

    context = f"epoch{epoch}"


    with open (f"{metrics_path}/score/epoch{epoch}_scores.pkl", "rb") as f:
        scores = pkl.load(f)


    with open (f"{metrics_path}/label/epoch{epoch}_labels.pkl", "rb") as f:
        labels = pkl.load(f)


    p_scores, n_scores = scores[np.where(labels == True)].astype(np.double), scores[np.where(labels == False)[0]].astype(np.double)

    scores = sorted(scores)
    n_scores = sorted(n_scores)
    p_scores = sorted(p_scores)


    result = []
    for score in scores:


    exit()

    Helper.generate_det_curve(p_scores, n_scores, f'{metrics_path}/pic/det_curve/{context}_det_curve.png')
    Helper.plot_roc(labels=labels, scores=scores, p_scores=p_scores, n_scores=n_scores, output_path=f"{metrics_path}/pic/roc/{context}_roc_curve.png")
    Helper.plot_probabilty_density(p_scores=p_scores, n_scores=n_scores, output_path=f"{metrics_path}/pic/PB/{context}_prob_den.png")
    Helper.plot_precision_recall(labels=labels, scores=scores, output_path=f"{metrics_path}/pic/PR/{context}_PR_curve.png")
