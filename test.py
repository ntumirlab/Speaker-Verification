#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pickle as pkl
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt
import time
import bob.measure
import bob.bio.gmm.algorithm.IVector as IVector

import numpy as np
from tqdm import tqdm
from eval_metrics import evaluate

from sre10Dataset_static import DeepSpeakerDataset
# from DeepSpeakerDataset_dynamic import DeepSpeakerDataset
from TestSet.ContinuousTestsetBySpeaker import ContinuousTestsetBySpeaker
from TestSet.ContinuousTestsetByUtt import ContinuousTestsetByUtt

from sre10_wav_reader import read_sre10_structure
from sklearn import preprocessing

from Model.FTDNN import FTDNN
from Model.TripletMarginLoss import PairwiseDistance, TripletMarginLoss
from Model.Resnet50_1d import ResNet50 as resnet
from audio_processing import toMFB, totensor, truncatedinput, tonormal, truncatedinputfromMFB, read_MFB, read_audio, mk_MFB
from Helper import Helper
import warnings
warnings.filterwarnings('ignore')


def check_and_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_gender(genders, scores, labels):
    sg_scores = []
    sg_labels = []
    dg_scores = []

    for i, gender in tqdm(enumerate(genders), desc="reading and sort..."):

        if gender == '1':
            sg_scores.append(scores[i])
            sg_labels.append(labels[i])
        else:
            dg_scores.append(scores[i])
    return np.array(sg_scores), np.array(sg_labels), np.array(dg_scores)


def output_eer(scores, labels, genders, spkrs_embeddings, spkr_labels, context, metrics_path):
    check_and_make_dir(f'{metrics_path}/score')
    check_and_make_dir(f'{metrics_path}/label')
    check_and_make_dir(f'{metrics_path}/pic/det_curve')
    check_and_make_dir(f"{metrics_path}/pic/PR")
    check_and_make_dir(f"{metrics_path}/pic/PB")
    check_and_make_dir(f"{metrics_path}/pic/roc")
    check_and_make_dir(f"{metrics_path}/pic/tsne")

    thr, fa, fr, Cprim, Cthr = Helper.cal_Cprimary(scores, labels)

    Helper.save(scores, f'{metrics_path}/score/{context}_scores.pkl')
    Helper.save(labels, f'{metrics_path}/label/{context}_labels.pkl')
    sg_scores, sg_labels, dg_scores = split_gender(genders, scores, labels)

    p_scores, n_scores = scores[np.where(labels == True)].astype(np.double), scores[np.where(labels == False)[0]].astype(np.double)
    Helper.save(p_scores, f'{metrics_path}/score/{context}_pscores.pkl')
    Helper.save(n_scores, f'{metrics_path}/score/{context}_nscores.pkl')

    sg_p_scores, sg_n_scores = sg_scores[np.where(sg_labels == True)].astype(np.double), sg_scores[np.where(sg_labels == False)[0]].astype(np.double)

    Helper.save(sg_p_scores, f'{metrics_path}/score/{context}_sg_p_scores.pkl')
    Helper.save(sg_n_scores, f'{metrics_path}/score/{context}_sg_n_scores.pkl')
    Helper.save(dg_scores, f'{metrics_path}/score/{context}_dg_scores.pkl')

    EER = Helper.generate_det_curve(p_scores, n_scores, f'{metrics_path}/pic/det_curve/{context}_det_curve.png')
    Helper.plot_roc(labels=labels, scores=scores, p_scores=p_scores, n_scores=n_scores, output_path=f"{metrics_path}/pic/roc/{context}_roc_curve.png")
    Helper.plot_probabilty_density(sg_p_scores=sg_p_scores, sg_n_scores=sg_n_scores, dg_scores=dg_scores, output_path=f"{metrics_path}/pic/PB/{context}_prob_den.png")
    Helper.plot_precision_recall(labels=labels, scores=scores, output_path=f"{metrics_path}/pic/PR/{context}_PR_curve.png")

    Helper.save(spkrs_embeddings, f"{metrics_path}/pic/tsne/spkrs_embeddings.pkl")
    Helper.save(spkr_labels, f"{metrics_path}/pic/tsne/spkr_labels.pkl")
    # if args.test_type == "test" and not args.test_by_epoch:
    #     Helper.plot_tsne(spkrs_embeddings, spkr_labels, feat_name=context, output_path=f"{metrics_path}/pic/tsne/{context}.png")
    return EER, Cprim

# def cosine_sim(model, probe):
#     return np.dot(model/np.linalg.norm(model), probe/np.linalg.norm(probe))
# Training settings

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
    # Model options
    parser.add_argument('--dataroot', type=str, default='/mnt/E/arthur.wang/aishell/dataset/utt/feature_type/test_type',
                        help='path to dataset')
    parser.add_argument('--test_pairs_path', type=str, default="/mnt/E/arthur.wang/aishell/aishell1/aishell1_test_type_list.txt",
                        help='path to pairs file')

    # Training options
    parser.add_argument('--embedding_size', type=int, default=64, metavar='ES',
                        help='Dimensionality of the embedding')

    parser.add_argument('--test_batch_size', type=int, default=64, metavar='BST',
                        help='input batch size for testing (default: 64)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--test_by_epoch', action='store_true', default=False)

    parser.add_argument('--res50', action='store_true', default=False)
    parser.add_argument('--byUtt', action='store_true', default=False)
    parser.add_argument('--oneD', action='store_true', default=True)
    parser.add_argument("--feature_type", type=str, default="mfcc", help="")
    parser.add_argument("--test_type", type=str, default="test", help="")
    parser.add_argument("--test_method", type=str, default="embed", help="")
    parser.add_argument("--dataset", type=str, default="aishell1", help="")
    parser.add_argument("--model_name", type=str, default="", help="")
    parser.add_argument("--epoch", type=str, default="50", help="")
    parser.add_argument("--total_epochs", type=int, default=50, help="")
    parser.add_argument("--ste", type=int, default=1, help="")

    args = parser.parse_args()
    args.metrics_path = f""



    if args.dataset == "aishell1":
        args.dataroot = f'/mnt/E/arthur.wang/aishell/aishell1/speaker_utt/{args.feature_type}/mix'
        args.test_pairs_dir = f"/mnt/E/arthur.wang/aishell/aishell1/test_pair/{args.feature_type}/{args.test_type}"
        args.meta_path = f"/mnt/E/arthur.wang/aishell/aishell1/resource_aishell/speaker.info"
        args.checkpoint_dir = f"/mnt/E/arthur.wang/Last_metrics/{args.model_name}/aishell1/{args.feature_type}"
        # args.checkpoint_dir = f"/mnt/E/arthur.wang/metrics/old/deepspeaker/aishell1/8_normalize_logfbank"
        args.num_of_class = 340
    elif args.dataset == "SRE10":
        args.dataroot = f'/mnt/E/arthur.wang/SRE10/speaker_utt/{args.feature_type}/{args.test_type}'
        args.test_pairs_dir = f"/mnt/E/arthur.wang/SRE10/test_pair/{args.feature_type}/{args.test_type}"
        args.checkpoint_dir = f"/mnt/E/arthur.wang/Last_metrics/{args.model_name}/SRE10/logfbank"
        args.num_of_class = 297
    elif args.dataset == "vox1":
        args.dataroot = f'/mnt/E/arthur.wang/vox1/speaker_utt/{args.feature_type}/mix'
        args.featureroot = f'/mnt/E/arthur.wang/vox1/{args.feature_type}/test'
        args.test_pairs_path = f"/mnt/E/arthur.wang/vox1/voxceleb1_test.txt"
        args.test_pairs_dir = f"/mnt/E/arthur.wang/vox1/test_pair/{args.feature_type}/{args.test_type}"
        args.meta_path = f"/mnt/E/arthur.wang/vox1/vox1_meta.csv"
        args.checkpoint_dir = f"/mnt/E/arthur.wang/Last_metrics/{args.model_name}/vox1/{args.feature_type}"
        # args.checkpoint_dir = f"/mnt/E/arthur.wang/metrics/old/deepspeaker/vox1/700_batch_2_soft+tri_8_normalize_logfbank"
        args.num_of_class = 1211
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        cudnn.benchmark = True
    args.device = f"cuda:{args.gpu_id}"

    args.spkr2gender = {}
    with open(args.meta_path, 'r') as ff:
        data = ff.readlines()
    if args.dataset == "vox1":
        data = data[1:]
        for line in data:
            line = line.strip().split('\t')
            args.spkr2gender[line[0]] = line[2]
    else:

        for line in data:
            line = line.strip().split()
            args.spkr2gender[f"S{line[0]}"] = line[1]
    args.cost_time_list = []
    args.eer_list = []
    args.Cprim_list = []

    return args


def main(args):
    np.random.seed(args.seed)
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_seconds_list = [1, 2, 4, 8, 16]

    if args.byUtt:
        test_seconds_list = [-1]

    test_seconds_list_str = [str(i) for i in test_seconds_list]



    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))


    if args.test_by_epoch:
        args.test_seconds = 1
        test_dir = ContinuousTestsetBySpeaker(utt_dir=args.dataroot, test_pairs_dir=args.test_pairs_dir, oneD=args.oneD, method=args.test_method, frames_num=args.test_seconds * 100)
        test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        for epoch in range(args.ste, args.total_epochs+1):

            model = resnet(embedding_size=args.embedding_size, num_classes=args.num_of_class)
            checkpoint = torch.load(f'{args.checkpoint_dir}/checkpoints/checkpoint_{epoch}.pth')
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.to(args.device)

            args.metrics_path = f"/mnt/E/arthur.wang/final_metrics/{args.model_name}/{args.dataset}/{args.feature_type}/{args.test_type}/test_by_epochs"

            test(test_loader, model, epoch, args.spkr2gender, args)
        pkl_dic = {"epochs": range(args.ste, args.total_epochs+1),
                   "eer": args.eer_list,
                   "Cprim": args.Cprim_list,
                   "cost time": args.cost_time_list}
        target_dir = f"Last_PIC/{args.dataset}/{args.model_name}/test_epochs/{args.test_type}"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        epoch_list_str = [str(i) for i in range(1, args.total_epochs+1)]
        with open(f"{target_dir}/result.txt", "w") as w:
            args.eer_list = [str(i) for i in args.eer_list]
            args.Cprim_list = [str(i) for i in args.Cprim_list]
            args.cost_time_list = [str(i) for i in args.cost_time_list]
            w.write(f"{' '.join(epoch_list_str)}\n")
            w.write(f"{' '.join(args.eer_list)}\n")
            w.write(f"{' '.join(args.Cprim_list)}\n")
            w.write(f"{' '.join(args.cost_time_list)}\n")

        with open(f"{target_dir}/result_dic.pkl", "wb") as w:
            pkl.dump(pkl_dic, w)
    else:
        if args.model_name == 'ivector':
            args.test_method = 'ivector'
            args.epoch = 'i'
            model = IVector(
                number_of_gaussians=256,
                gmm_training_iterations=20,
                use_whitening=False,
                use_wccn=False,
                use_lda=True,
                lda_dim=100,
                subspace_dimension_of_t=300,
                tv_training_iterations=5,
                use_plda=True,
                plda_dim_F=50,
                plda_dim_G=50,
                plda_training_iterations=50
            )
            model.load_projector('aishell1-plda-lda.hdf5')
        else:
            model = resnet(embedding_size=args.embedding_size, num_classes=args.num_of_class)
            checkpoint = torch.load(f'{args.checkpoint_dir}/checkpoints/checkpoint_{args.epoch}.pth')
            model.load_state_dict(checkpoint['state_dict'])
            # model.load_state_dict(torch.load(f'{args.checkpoint_dir}/checkpoints/checkpoint_{args.epoch}.pth'))

            if args.cuda:
                model.to(args.device)
        for test_seconds in test_seconds_list:
            args.test_seconds = test_seconds
            args.test_batch_size = int(240/test_seconds)

            args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/{args.model_name}/{args.dataset}/{args.feature_type}/{args.test_type}/epoch_{args.epoch}/{args.test_seconds}_seconds"
            if args.byUtt:
                test_loader = ContinuousTestsetByUtt(feature_dir=args.featureroot, pairs_path=args.test_pairs_path, spkr2gender=args.spkr2gender, normalize=True, std=True)
                args.test_batch_size = 16
            else:
                test_dir = ContinuousTestsetBySpeaker(utt_dir=args.dataroot, test_pairs_dir=args.test_pairs_dir, oneD=args.oneD, method=args.test_method, frames_num=args.test_seconds * 100)

                test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, **kwargs)

            test(test_loader, model, args.epoch, args.spkr2gender, args)

        pkl_dic = {"test_seconds": test_seconds_list,
                "eer": args.eer_list,
                "Cprim": args.Cprim_list,
                "cost time": args.cost_time_list}
        if not os.path.exists(f"Last_PIC/{args.dataset}/{args.model_name}/epoch_{args.epoch}/{args.test_type}"):
            os.makedirs(f"Last_PIC/{args.dataset}/{args.model_name}/epoch_{args.epoch}/{args.test_type}")

        with open(f"Last_PIC/{args.dataset}/{args.model_name}/epoch_{args.epoch}/{args.test_type}/result.txt", "w") as w:
            args.eer_list = [str(i) for i in args.eer_list]
            args.Cprim_list = [str(i) for i in args.Cprim_list]
            args.cost_time_list = [str(i) for i in args.cost_time_list]
            w.write(f"{' '.join(test_seconds_list_str)}\n")
            w.write(f"{' '.join(args.eer_list)}\n")
            w.write(f"{' '.join(args.Cprim_list)}\n")
            w.write(f"{' '.join(args.cost_time_list)}\n")

        with open(f"Last_PIC/{args.dataset}/{args.model_name}/epoch_{args.epoch}/{args.test_type}/result_dic.pkl", "wb") as w:
            pkl.dump(pkl_dic, w)


def test(test_loader, model, epoch, spkr2gender, args):


    # switch to evaluate mode
    print("testing")
    model.eval()
    criterion = nn.CosineSimilarity(dim=1, eps=1e-6).to(args.device)
    # criterion = PairwiseDistance(2)

    with torch.no_grad():
        labels, distances, genders, spkrs_embeddings, spkr_labels = [], [], [], [], []

        pbar = tqdm(enumerate(test_loader))
        time_start = time.time()
        # for batch_idx, (data_a, data_p, label, gender, length_a, length_p) in pbar:
        # for batch_idx,  in pbar:
        for batch_idx, datas in pbar:

            if args.test_method == "avgembed":
                (data_a, data_p, label, gender, length_a, length_p) = datas
            else:
                (data_a, data_p, label, gender, spkr_1, spkr_2) = datas

            if args.cuda:
                data_a, data_p = data_a.to(args.device), data_p.to(args.device)
            if args.byUtt:
                data_a = data_a.unsqueeze(0)
                data_p = data_p.unsqueeze(0)
            out_a, out_p = model(data_a), model(data_p)

            spkrs_embeddings.append(out_a.data.cpu().numpy())
            spkrs_embeddings.append(out_p.data.cpu().numpy())

            spkr_labels.append(spkr_1)
            spkr_labels.append(spkr_2)
            if args.test_method == "avgembed":
                dists = []
                start_a = 0
                start_b = 0
                for segment_a, segment_p in zip(length_a, length_p):

                    avg_embedding_a = torch.mean(out_a[start_a:start_a + segment_a], 0).unsqueeze(0)
                    avg_embedding_p = torch.mean(out_p[start_b:start_b + segment_p], 0).unsqueeze(0)
                    di = cos(avg_embedding_a, avg_embedding_p)

                    dists.append(di.data.cpu().numpy()[0])
                    start_a += segment_a
                    start_b += segment_p
            else:
                # dists = cos(out_a, out_p).data.cpu().numpy()
                dists = criterion.forward(out_a, out_p).data.cpu().numpy()

            distances.append(dists)
            labels.append(label)
            genders.append(gender)
            pbar.set_description('Test time: {}, Epoch: {} [{}/{} ({:.0f}%)]'.format(
                args.test_seconds, epoch, batch_idx, 45000,
                100. * batch_idx / len(test_loader)))

        time_end = time.time()
        if args.byUtt:
            labels = np.array(labels)
            genders = np.array(genders)
            spkrs_embeddings = np.array(spkrs_embeddings)
            spkr_labels = np.array(spkr_labels)
        else:
            labels = np.array([sublabel for label in labels for sublabel in label])
            genders = np.array([subgender for gender in genders for subgender in gender])
            spkrs_embeddings = np.array([subembed for embed in spkrs_embeddings for subembed in embed])
            spkr_labels = np.array([sublabel for label in spkr_labels for sublabel in label])

        # print(spkr_labels)
        spkrs_embeddings_dic = {}
        for idx, embed in enumerate(spkrs_embeddings):
            if spkr_labels[idx] not in spkrs_embeddings_dic:
                spkrs_embeddings_dic[spkr_labels[idx]] = []
            spkrs_embeddings_dic[spkr_labels[idx]].append(embed)

        spkrs_embeddings = {"M":[], "F":[]}
        spkr_labels = {"M":[], "F":[]}

        for idx, (spkr, embedings) in enumerate(spkrs_embeddings_dic.items()):
            spkrs_embeddings[spkr2gender[spkr]] += embedings[:20]
            spkr_labels[spkr2gender[spkr]] += [idx]*20

        spkrs_embeddings["F"] = spkrs_embeddings["F"]
        spkrs_embeddings["M"] = spkrs_embeddings["M"]
        spkr_labels["F"] = np.array(spkr_labels["F"])
        spkr_labels["M"] = np.array(spkr_labels["M"])

        # markers = np.array(markers)
        # if args.test_method != "avgembed":
        #     distances = np.array([subdist * -1 for dist in distances for subdist in dist], dtype=np.float64)
        # else:
        distances = np.array([subdist for dist in distances for subdist in dist], dtype=np.float64)

        # for i, di in enumerate(distances):
        #     if di < -100.:
        #         distances[i] = -100.
        # distances = distances[np.isnan(distances)] = 0
        # distances = distances[np.isinf(distances)] = 0

        print("output eer")
        eer, cprim = output_eer(distances, labels, genders, spkrs_embeddings, spkr_labels, f"epoch{epoch}_{args.test_seconds}_seconds", args.metrics_path)
        args.eer_list.append(eer)
        args.Cprim_list.append(cprim)
        args.cost_time_list.append((time_end - time_start)/45000)


def test_ivector(test_loader, model):

    # def cosine_sim(model, probe):
    #     return np.dot(model/np.linalg.norm(model), probe/np.linalg.norm(probe))
    # switch to evaluate mode
    print("testing ivector")

    labels, distances, genders = [], [], []

    pbar = tqdm(enumerate(test_loader))
    time_start = time.time()

    for batch_idx, (data_a, data_p, label, gender) in pbar:

        data_a = data_a[0].data.numpy()
        data_p = data_p[0].data.numpy()
        label = label[0].data.numpy()
        gender = gender[0]
        iv_1 = model.project(np.array(data_a))
        iv_2 = model.project(np.array(data_p))
        di = cosine_sim(iv_1, iv_2)

        distances.append(di)


        # print(embedding_a[0].shape)
        # print(embedding_p.shape)
        # embedding_a = torch.stack(embedding_a)
        # embedding_p = torch.stack(embedding_p)

        # dists = l2_dist.forward(embedding_a, embedding_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        # dists = dists.data.cpu().numpy()
        labels.append(label)
        genders.append(gender)
        if batch_idx % args.log_interval == 0:
            pbar.set_description('[{}/{} ({:.0f}%)]'.format(
                batch_idx, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))
    time_end = time.time()
    args.cost_time_list.append((time_end - time_start)/45000)

    print("output eer")
    eer, cprim = output_eer(np.array(distances), np.array(labels), genders, f"{args.test_seconds}_seconds", args.metrics_path, args.test_pairs_path)
    args.eer_list.append(eer)
    args.Cprim_list.append(cprim)



if __name__ == '__main__':
    args = get_args()
    main(args)
