#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import time
import numpy as np
from tqdm import tqdm
from TrainSet.ClassificationDataset import ClassificationDataset
from TestSet.ContinuousTestsetBySpeaker import ContinuousTestsetBySpeaker
from TrainSet.TripletDataset import TripletDataset
from Model.TripletMarginLoss import TripletMarginLoss, PairwiseDistance
from Model.TripletCosMarginLoss import TripletCosMarginLoss
from Model.AngularTripletMarginLoss import AngularTripletMarginLoss
from Model.Resnet50_1d import ResNet50 as resnet
from Model.AngularTripletCenterMarginLoss import AngularTripletCenterMarginLoss
from Model.AAML import AngularPenaltySMLoss
from Helper import Helper
from Model.GE2ELoss import GE2ELoss
from TrainSet.GE2EDataset import GE2EDataset
import warnings
warnings.filterwarnings('ignore')


def check_and_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Speaker Verification')
    # Model options
    parser.add_argument('--valid_dir', type=str, default='/mnt/E/arthur.wang/aishell/aishell1/feature_type',
                        help='path to dataset')
    parser.add_argument('--test_pairs_path', type=str, default="/mnt/E/arthur.wang/aishell/aishell1/aishell1_test_list.txt",
                        help='path to pairs file')
    parser.add_argument('--checkpoint_dir', default='./data/checkpoints',
                        help='folder to output model checkpoints')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 10)')
    # Training options
    parser.add_argument('--embedding_size', type=int, default=64, metavar='ES',
                        help='Dimensionality of the embedding')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='BST',
                        help='input batch size for testing (default: 64)')

    parser.add_argument('--test_input_seconds', type=int, default=-1)
    parser.add_argument('--n_triplets', type=int, default=100000, metavar='N',
                        help='how many triplets will generate from the dataset')

    parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                        help='the margin value for the triplet loss function (default: 1.0')
    parser.add_argument('--angular_triplet_margin', type=float, default=0.4)
    parser.add_argument('--angular_softmax_margin', type=float, default=0.2)
    parser.add_argument('--min_softmax_epoch', type=int, default=0, metavar='MINEPOCH',
                        help='minimum epoch for initial parameter using softmax (default: 2')

    parser.add_argument('--loss_ratio', type=float, default=1.0, metavar='LOSSRATIO',
                        help='the ratio softmax loss - triplet loss (default: 2.0')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.125)')
    parser.add_argument('--lr_decay', default=1e-4, type=float, metavar='LRD',
                        help='learning rate decay ratio (default: 1e-4')
    parser.add_argument('--wd', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 0.0)')
    parser.add_argument('--optimizer', default='adam', type=str,
                        metavar='OPT', help='The optimizer to use (default: Adagrad)')
    # Device options
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log_interval', type=int, default=1, metavar='LI',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--only_test', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--feature_type", type=str, default="mfcc", help="")

    parser.add_argument('--loss_function', default='', type=str, help='loss function', choices=["SM", "ATL", "TE", "GE2E", "AS", "ATCL", "TCL", "ATCLGE2E"])
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_opt', action='store_true', default=False)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument("--resume_model", type=str, default="aishell1", help="")
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--ge2e_pretrain', action='store_true', default=False)
    parser.add_argument('--as_pretrain', action='store_true', default=False)
    parser.add_argument('--atl_pretrain', action='store_true', default=False)
    parser.add_argument('--res50', action='store_true', default=False)
    parser.add_argument('--oneD', action='store_true', default=True)
    parser.add_argument('--random_window', action='store_true', default=False)
    parser.add_argument('--speaker_num', type=int, default=60)
    parser.add_argument('--utterance_num', type=int, default=10)
    parser.add_argument('--center_step', type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="aishell1", help="")
    parser.add_argument('--pretrain_epoch', type=int, default=95)
    parser.add_argument('--softmax_iterations', type=int, default=500000)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--std', action='store_true', default=False)
    parser.add_argument('--static', action='store_true', default=False)
    parser.add_argument('--mix', action='store_true', default=False)
    parser.add_argument('--mix_as', action='store_true', default=False)

    args = parser.parse_args()
    if args.dataset == "aishell1":
        args.speaker_utt_dir = f"/mnt/E/arthur.wang/aishell/{args.dataset}/speaker_utt/{args.feature_type}/train"
        args.utt_dir = f'/mnt/E/arthur.wang/aishell/aishell1/speaker_utt/{args.feature_type}'
        args.test_pairs_dir = f"/mnt/E/arthur.wang/aishell/aishell1/test_pair/{args.feature_type}"
    else:
        args.speaker_utt_dir = f"/mnt/E/arthur.wang/{args.dataset}/speaker_utt/{args.feature_type}/train"
        args.utt_dir = f'/mnt/E/arthur.wang/{args.dataset}/speaker_utt/{args.feature_type}'
        args.test_pairs_dir = f"/mnt/E/arthur.wang/{args.dataset}/test_pair/{args.feature_type}"

    args.valid_dir = args.valid_dir.replace("feature_type", args.feature_type)
    return args


def main(args):
    np.random.seed(args.seed)
    args.cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    args.device = f"cuda:{args.gpu_id}"



    valid_dir = ContinuousTestsetBySpeaker(utt_dir=f"{args.utt_dir}/valid", test_pairs_dir=f"{args.test_pairs_dir}/valid", oneD=args.oneD, method="eval", frames_num=100, normalize=args.normalize, std=args.std)
    valid_loader = torch.utils.data.DataLoader(valid_dir, batch_size=240, shuffle=False, **kwargs)
    test_dir = ContinuousTestsetBySpeaker(utt_dir=f"{args.utt_dir}/test", test_pairs_dir=f"{args.test_pairs_dir}/test", oneD=args.oneD, method="eval", frames_num=100, normalize=args.normalize, std=args.std)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=240, shuffle=False, **kwargs)
    if args.pretrain:
        pretrain_label = f"_SoftmaxPretrain_{args.pretrain_epoch}"
    elif args.ge2e_pretrain:
        pretrain_label = f"_GE2EPretrain_{args.pretrain_epoch}"
    elif args.as_pretrain:
        pretrain_label = f"_ASPretrain_{args.pretrain_epoch}"
    elif args.atl_pretrain:
        pretrain_label = f"_ATLPretrain_{args.pretrain_epoch}"
    else:
        pretrain_label = ""
        # args.lr = 0.01
    if args.normalize:
        normalize_label = "_normalize"
    else:
        normalize_label = ""

    if args.random_window:
        ramdom_label = "_random"
    else:
        ramdom_label = ""



    if args.mix:
        mix_label = f"_mix{args.loss_ratio}_SM"
    elif args.mix_as:
        mix_label = f"_mix{args.loss_ratio}_AS_{args.angular_softmax_margin}"
    else:
        mix_label = ""

    if args.loss_function == "ATL":
        train_loader = TripletDataset(utt_dir=args.speaker_utt_dir, n_triplets=args.n_triplets, oneD=args.oneD, random_window=args.random_window, normalize=args.normalize, std=args.std)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/AngularTriplet_{args.angular_triplet_margin}{mix_label}{pretrain_label}{normalize_label}/{args.dataset}/{args.feature_type}"
    elif args.loss_function == "ATCL":
        train_loader = GE2EDataset(utt_dir=args.speaker_utt_dir, speaker_num=args.speaker_num, utterance_num=args.utterance_num, step=args.center_step, oneD=args.oneD, random_window=args.random_window)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/ATCL_SGD_{args.angular_triplet_margin}{mix_label}{pretrain_label}{normalize_label}/{args.dataset}/{args.feature_type}"
    elif args.loss_function == "TE":
        train_loader = TripletDataset(utt_dir=args.speaker_utt_dir, n_triplets=args.n_triplets, oneD=args.oneD, random_window=args.random_window)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/TriEuc_{args.margin}{pretrain_label}{normalize_label}/{args.dataset}/{args.feature_type}"
    elif args.loss_function == "GE2E":
        train_loader = GE2EDataset(utt_dir=args.speaker_utt_dir, speaker_num=args.speaker_num, utterance_num=args.utterance_num, step=args.center_step, oneD=args.oneD, random_window=args.random_window)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/GE2E{mix_label}{pretrain_label}{normalize_label}/{args.dataset}/{args.feature_type}"
    elif args.loss_function == "SM":
        train_loader = ClassificationDataset(utt_dir=args.speaker_utt_dir, iterations=args.softmax_iterations, random_window=args.random_window, oneD=args.oneD, normalize=args.normalize, std=args.std, static=args.static)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/Softmax_Sample_{normalize_label}{ramdom_label}/{args.dataset}/{args.feature_type}"
        # args.optimizer == "rmsprop"
    elif args.loss_function == "AS":
        train_loader = ClassificationDataset(utt_dir=args.speaker_utt_dir, iterations=args.softmax_iterations, random_window=args.random_window, oneD=args.oneD, normalize=args.normalize, std=args.std)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/A-Softmax_{args.angular_softmax_margin}{pretrain_label}{normalize_label}/{args.dataset}/{args.feature_type}"
    elif args.loss_function == "ATCLGE2E":
        train_loader = GE2EDataset(utt_dir=args.speaker_utt_dir, speaker_num=args.speaker_num, utterance_num=args.utterance_num, step=args.center_step, oneD=args.oneD, random_window=args.random_window)
        args.metrics_path = f"/mnt/E/arthur.wang/Last_metrics/GE2E_ATCL_{args.angular_triplet_margin}{mix_label}{pretrain_label}{normalize_label}/{args.dataset}/{args.feature_type}"

    check_and_make_dir(f"{args.metrics_path}/loss")
    check_and_make_dir(f"{args.metrics_path}/checkpoints")
    check_and_make_dir(f"{args.metrics_path}/eval")

    num_of_class = train_loader.n_classes
    print('\nNumber of Classes:\n{}\n'.format(train_loader.n_classes))
    model = resnet(embedding_size=args.embedding_size, num_classes=num_of_class)

    if args.cuda:
        cudnn.benchmark = True
        model.to(args.device)
    optimizer = create_optimizer(model, args.lr)

    valid_eer = []
    test_eer = []

    if args.resume:
        checkpoint = torch.load(f'/mnt/E/arthur.wang/Last_metrics/{args.resume_model}/{args.dataset}/{args.feature_type}/checkpoints/checkpoint_{args.resume_epoch}.pth')
        model.load_state_dict(checkpoint['state_dict'])
        valid_eer = Helper.load(f"{args.metrics_path}/eval/valid_eer.pkl")[:args.resume_epoch]
        test_eer = Helper.load(f"{args.metrics_path}/eval/test_eer.pkl")[:args.resume_epoch]
        if args.resume_opt:
            optimizer.load_state_dict(checkpoint['optimizer'])
    elif args.pretrain:
        checkpoint = torch.load(f'/mnt/E/arthur.wang/Last_metrics/Softmax{normalize_label}/{args.dataset}/{args.feature_type}/checkpoints/checkpoint_{args.pretrain_epoch}.pth')
        model.load_state_dict(checkpoint['state_dict'])
    elif args.ge2e_pretrain:
        checkpoint = torch.load(f'/mnt/E/arthur.wang/Last_metrics/GE2E{normalize_label}/{args.dataset}/{args.feature_type}/checkpoints/checkpoint_{args.pretrain_epoch}.pth')
        model.load_state_dict(checkpoint['state_dict'])
    elif args.as_pretrain:
        checkpoint = torch.load(f'/mnt/E/arthur.wang/Last_metrics/A-Softmax_0.2{normalize_label}/{args.dataset}/{args.feature_type}/checkpoints/checkpoint_{args.pretrain_epoch}.pth')
        model.load_state_dict(checkpoint['state_dict'])
    elif args.atl_pretrain:
        checkpoint = torch.load(f'/mnt/E/arthur.wang/Last_metrics/AngularTriplet_0.5_mix1.0{normalize_label}/{args.dataset}/{args.feature_type}/checkpoints/checkpoint_{args.pretrain_epoch}.pth')
        model.load_state_dict(checkpoint['state_dict'])
    # if args.loss_function == "GE2E":
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 6, gamma=0.1, last_epoch=-1)
    # elif args.loss_function == "SM":
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    # else:
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 6, gamma=0.1, last_epoch=-1)

    print('\nparsed options:\n{}\n'.format(vars(args)))

    if args.resume:
        start = args.resume_epoch + 1
    else:
        start = args.start_epoch
    end = start + args.epochs

    train_start_time = time.time()

    for epoch in range(start, end):
        # lr_scheduler.step()

        if args.loss_function in ["GE2E", "ATCL", "ATCLGE2E"]:
            train_center(train_loader, model, optimizer, epoch, args)
        elif args.loss_function in ["ATL", "TE"]:
            train_triplet(train_loader, model, optimizer, epoch, args)
        elif args.loss_function in ["SM", "AS"]:
            train_softmax(train_loader, model, optimizer, epoch, args)
        eval(model, test_loader, test_eer, epoch, args, "test")
        eval(model, valid_loader, valid_eer, epoch, args, "valid")
        #break;
    print("Training time:", time.time()-train_start_time)


def eval(model, test_loader, eer_list, epoch, args, test_type):
    print(f"evaluating {test_type}...")
    model.eval()

    if args.loss_function == "TE":
        sign = -1
        criterion = PairwiseDistance(2)
    else:
        sign = 1
        criterion = nn.CosineSimilarity(dim=1, eps=1e-6).to(args.device)

    with torch.no_grad():
        labels, distances, genders = [], [], []

        pbar = tqdm(enumerate(test_loader))
        # for batch_idx, (data_a, data_p, label, gender, length_a, length_p) in pbar:
        for batch_idx, (data_a, data_p, label, gender, _, _) in pbar:

            data_a, data_p = data_a.to(args.device), data_p.to(args.device)

            out_a, out_p = model(data_a), model(data_p)

            dists = criterion.forward(out_a, out_p).data.cpu().numpy()

            distances.append(dists)
            labels.append(label)
            genders.append(gender)
            pbar.set_description('Test time: {}, Epoch: {} [{}/{} ({:.0f}%)]'.format(
                1, epoch, batch_idx, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))
        labels = np.array([sublabel for label in labels for sublabel in label])
        genders = np.array([subgender for gender in genders for subgender in gender])

        distances = np.array([subdist * sign for dist in distances for subdist in dist], dtype=np.float64)
        p_scores, n_scores = distances[np.where(labels == True)].astype(np.double), distances[np.where(labels == False)[0]].astype(np.double)
        far, frr, eer = Helper.cal_eer(p_scores=p_scores, n_scores=n_scores)
        eer_list.append(eer)
        Helper.save(eer_list, f"{args.metrics_path}/eval/{test_type}_eer.pkl")
        print('\33[91mEval: EER: {:.8f}\n\33[0m'.format(eer))


def train_triplet(train_loader, model, optimizer, epoch, args):
    # switch to train mode
    print("Training...")
    model.train()

    loss_list = []
    pbar = tqdm(enumerate(train_loader))

    if args.loss_function == "TE":
        criterion = TripletMarginLoss(args.margin)
    elif args.loss_function == "ATL":
        criterion = AngularTripletMarginLoss(margin=args.angular_triplet_margin)

    if args.mix:
        softmax_criterion = nn.CrossEntropyLoss()
    elif args.mix_as:
        softmax_criterion = AngularPenaltySMLoss(m=args.angular_softmax_margin)

    for batch_idx, (data_a, data_p, data_n, label_p, label_n) in pbar:

        data_a, data_p, data_n = data_a.to(args.device), data_p.to(args.device), data_n.to(args.device)
        label_p, label_n = label_p.to(args.device), label_n.to(args.device)

        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

        triplet_loss = criterion.forward(out_a, out_p, out_n)

        if args.mix or args.mix_as:
            cls_a, cls_p, cls_n = model.forward_classifier(data_a), model.forward_classifier(data_p), model.forward_classifier(data_n)
            softmax_loss = softmax_criterion(cls_a, label_p) + softmax_criterion(cls_p, label_p) + softmax_criterion(cls_n, label_n)
            loss = softmax_loss + args.loss_ratio * triplet_loss
        else:
            loss = triplet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            if args.mix or args.mix_as:
                pbar.set_description(f'Train Epoch: {epoch:3d} [{batch_idx * args.batch_size:8d}/{len(train_loader):8d} ({100. * batch_idx / (len(train_loader)/args.batch_size):3.0f}%)]\tTriplet Loss: {triplet_loss.item():.6f}\t Softmax Loss {softmax_loss.item():.6f}')
            else:
                pbar.set_description(f'Train Epoch: {epoch:3d} [{batch_idx * args.batch_size:8d}/{len(train_loader):8d} ({100. * batch_idx / (len(train_loader)/args.batch_size):3.0f}%)]\tLoss: {loss.item():.6f}')
        loss_list.append(loss.item())
    Helper.plot_loss(loss_list, f"{args.metrics_path}/loss/loss_{epoch}.png")
    Helper.save(loss_list, f"{args.metrics_path}/loss/loss_{epoch}.pkl")
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, f'{args.metrics_path}/checkpoints/checkpoint_{epoch}.pth')


def train_center(train_loader, model, optimizer, epoch, args):
    # switch to train mode
    print("Training...")
    model.train()
    if args.loss_function == "GE2E":
        criterion = GE2ELoss().to(args.device)
    elif args.loss_function == "ATCL":
        criterion = AngularTripletCenterMarginLoss(device=args.device, margin=args.angular_triplet_margin, spkr_num=args.speaker_num, utt_num=args.utterance_num)
    elif args.loss_function == "ATCLGE2E":
        criterion = GE2ELoss().to(args.device)
        atcl_criterion = AngularTripletCenterMarginLoss(device=args.device, margin=args.angular_triplet_margin, spkr_num=args.speaker_num, utt_num=args.utterance_num)

    if args.mix:
        softmax_criterion = nn.CrossEntropyLoss()
    elif args.mix_as:
        softmax_criterion = AngularPenaltySMLoss(m=args.angular_softmax_margin)

    loss_list = []
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, label) in pbar:
        #print("on training{}".format(epoch))

        data_a = data_a.to(args.device)
        label = label.to(args.device)
        data_a = data_a.resize_(args.speaker_num * args.utterance_num, 39, data_a.shape[-1])
        # compute output
        out_a = model(data_a)
        out_a = out_a.view(args.speaker_num, args.utterance_num, 64)

        center_loss = criterion.forward(out_a)

        if args.mix or args.mix_as:
            cls_a = model.forward_classifier(data_a)
            softmax_loss = softmax_criterion(cls_a, label)
            loss = softmax_loss + args.loss_ratio * center_loss
        elif args.loss_function == "ATCLGE2E":
            out_b = model(data_a)
            out_b = out_b.view(args.speaker_num, args.utterance_num, 64)
            atcl_loss = atcl_criterion.forward(out_b)
            loss = center_loss + args.loss_ratio * atcl_loss
        else:
            loss = center_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            if args.mix or args.mix_as:
                pbar.set_description(f'Train Epoch: {epoch:3d} {batch_idx}/{args.center_step}\tCenter Loss: {center_loss.item():.6f}\t SoftmaxLoss {softmax_loss:.6f}')
            elif args.loss_function == "ATCLGE2E":
                pbar.set_description(f'Train Epoch: {epoch:3d} {batch_idx}/{args.center_step}\tGE2E Loss: {center_loss.item():.6f}\t ATCLoss {atcl_loss:.6f}')
            else:
                pbar.set_description(f'Train Epoch: {epoch:3d} {batch_idx}/{args.center_step}\tLoss: {loss.item():.6f}')

        loss_list.append(loss.item())
    Helper.plot_loss(loss_list, f"{args.metrics_path}/loss/loss_{epoch}.png")
    Helper.save(loss_list, f"{args.metrics_path}/loss/loss_{epoch}.pkl")
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, f'{args.metrics_path}/checkpoints/checkpoint_{epoch}.pth')

def train_softmax(train_loader, model, optimizer, epoch, args):
    # switch to train mode
    print("Training...")
    model.train()
    if args.loss_function == "SM":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == "AS":
        criterion = AngularPenaltySMLoss(m=args.angular_softmax_margin)
    criterion = criterion.to(args.device)
    loss_list = []
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, label_p) in pbar:
        #print("on training{}".format(epoch))
        # data_a = data_a.squeeze(1).transpose(2, 1)
        data_a = data_a.to(args.device)
        label_p = label_p.to(args.device)

        cls_a = model.forward_classifier(data_a).to(args.device)

        loss = criterion(cls_a, label_p)

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                f'Train Epoch: {epoch:3d} [{batch_idx}/{args.softmax_iterations}\tLoss: {loss.item():.6f}')
        loss_list.append(loss.item())
    Helper.plot_loss(loss_list, f"{args.metrics_path}/loss/loss_{epoch}.png")
    Helper.save(loss_list, f"{args.metrics_path}/loss/loss_{epoch}.pkl")
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, f'{args.metrics_path}/checkpoints/checkpoint_{epoch}.pth')


# def train_trip_pretrain(train_loader, model, optimizer, epoch, args):

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=args.wd)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    args = get_args()
    main(args)
