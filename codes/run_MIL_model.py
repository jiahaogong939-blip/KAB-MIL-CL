import numpy as np
import torch
from genomicWSIdataset import GenomicWSIDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import check_dir, setup_seed, get_logger
import torch.nn as nn
from models import MILAttenFusion
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR, \
    CosineAnnealingLR, SequentialLR, ConstantLR, ExponentialLR
import argparse
from evaluation import eval_model
import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import os, random
import json
import pandas as pd
from sklearn import preprocessing
import subprocess
import torch.nn.functional as F
from sklearn.cluster import KMeans

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#超参数设置
def params_setup(dropout):
    params = dict()
    params["in_chans"] = [200, 16]
    params["dims"] = [512, 512]
    params["out_chans"] = [16, 8]
    params["heads"] = [8, 4]
    params["dropout"] = dropout
    return params

#记录打印参数
def logger_initial(logger, params):
    logger.info('params["in_chans"]=[%d, %d]' % (params["in_chans"][0], params["in_chans"][1]))
    logger.info('params["dims"]=[%d, %d]' % (params["dims"][0], params["dims"][1]))
    logger.info('params["out_chans"]=[%d, %d]' % (params["out_chans"][0], params["out_chans"][1]))
    logger.info('params["heads"]=[%d, %d]' %(params["heads"][0], params["heads"][1]))
    logger.info('params["dropout"]=[%.1f, %.1f, %.1f]' %(params["dropout"][0], params["dropout"][1], params["dropout"][2]))
    return logger


class savePath():
    def __init__(self, proj, time_tab):
        partient_dir = "{0}/{1}/".format("TrainProcess", proj)
        self.model_path = partient_dir + time_tab + "/" + "model" + "/"
        check_dir(self.model_path)
        self.embed_path = partient_dir + time_tab + "/" + "embed" + "/"
        check_dir(self.embed_path)
        self.record_path = partient_dir + time_tab + "/" + "record" + "/"
        check_dir(self.record_path)
        self.log_path = partient_dir + time_tab + "/" + time_tab + ".log"
        self.writer_path = partient_dir + time_tab + "/" + "tensorboard" + "/" + time_tab
        check_dir(self.writer_path)
        self.argument_path = partient_dir + time_tab + "/" + time_tab + ".json"
        #tnse


class TrainingConfig():
    #定义了一个 TrainingConfig 类。这个类用于初始化一些与训练配置相关的参数，并将它们存储为对象的属性。
    def __init__(self, logger, writer, save_path, args):
        #设置类的属性
        self.logger = logger
        self.writer = writer
        self.model_path = save_path.model_path
        self.embed_path = save_path.embed_path
        self.record_path = save_path.record_path
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.lam = args.lam
        self.scheduler = args.scheduler
        self.savedisk = args.savedisk
        self.smoothing = args.smoothing
        self.tau = args.tau
        ##
        #取值一般是在0到1，现在已取0.07，0.1
        self.temperature = 0.55
        #KR数据不取0.6（结果不好）


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    # F.normalize()意思是对输入的张量进行标准化，即将张量的每个分量除以其范数。
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def info_nce(query, positive_key, negative_keys=None, temperature=0.1,
             reduction='mean', negative_mode='unpaired'):
    if query.dim() != 2:
        raise ValueError('query must be 2D tensor')
    if positive_key.dim() != 2:
        raise ValueError('positive_key must be 2D tensor')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError('negative_keys must be 2D tensor for negative_mode=unpaired')
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError('negative_keys must be 3D tensor for negative_mode=paired')
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ transpose(positive_key)
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


# PCA去噪模块：每个患者（bag）单独处理
def pca_denoise_patient(features, energy_ratio=0.95, alpha=1.0):
    # features: torch tensor, shape [num_instances, feature_dim]
    feat_np = features.detach().cpu().numpy()
    mu = np.mean(feat_np, axis=0, keepdims=True)
    feat_centered = feat_np - mu
    cov = np.cov(feat_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    total_energy = np.sum(eigvals)
    energy_cumsum = np.cumsum(eigvals) / total_energy
    Ni = np.searchsorted(energy_cumsum, energy_ratio) + 1
    tau = alpha * np.median(eigvals[Ni:])
    eigvals_shrink = np.array([eig if i < Ni else max(eig - tau, 0) for i, eig in enumerate(eigvals)])
    feat_denoised = feat_centered @ eigvecs @ np.diag(eigvals_shrink) @ eigvecs.T + mu
    return torch.tensor(feat_denoised, device=features.device, dtype=features.dtype)


def train(predict_model, train_iter, test_iter, CLASSES, optimizer, scheduler, args):
    num_epochs = args.num_epochs
    writer = args.writer
    logger = args.logger
    device = args.device
    
    predict_model = predict_model.float().to(device)
    prev_loss = float('inf')
    max_retries = 5
    round = 0  # To track scheduler steps

    for epoch in range(num_epochs):
        retries = 0
        while retries < max_retries:
            # Save model and optimizer states before the epoch
            state_before_epoch = predict_model.state_dict().copy()
            optimizer_state_before_epoch = optimizer.state_dict().copy()
            
            predict_model.train()
            loss_, ce_loss_, nce_loss_ = 0.0, 0.0, 0.0
            batch_count = 0.0

            for X, y, pid, _ in train_iter:
                X = X.float().to(device)
                y = y.to(device)

                # 前向传播
                featfeat_tr, linearprob_tr = predict_model(X)

                # PCA特征去噪（每个患者单独处理）
                featfeat_tr = pca_denoise_patient(featfeat_tr, energy_ratio=0.95, alpha=1.0)

                CE_loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=args.smoothing)
                ce_loss = CE_loss(linearprob_tr, y)

                # KMeans clustering
                kmeans = KMeans(n_clusters=10, random_state=0).fit(featfeat_tr.detach().cpu().numpy())
                cluster_labels = torch.tensor(kmeans.labels_).to(device)

                # Create positive and negative keys
                positive_keys, negative_keys = [], []
                for i in range(X.size(0)):
                    current_cluster = cluster_labels[i].item()
                    pos_indices = (cluster_labels == current_cluster).nonzero(as_tuple=True)[0]
                    pos_sample_idx = random.choice(pos_indices.tolist())
                    positive_keys.append(featfeat_tr[pos_sample_idx].unsqueeze(0))
                    
                    negative_indices = (cluster_labels != current_cluster).nonzero(as_tuple=True)[0]
                    neg_sample_idx = random.choice(negative_indices.tolist())
                    negative_keys.append(featfeat_tr[neg_sample_idx].unsqueeze(0))

                positive_keys = torch.cat(positive_keys, dim=0).to(device)
                negative_keys = torch.cat(negative_keys, dim=0).to(device)

                info_nce_loss = InfoNCE(temperature=args.temperature, reduction='mean', negative_mode='unpaired')
                nce_loss_ = info_nce_loss(featfeat_tr, positive_keys, negative_keys)
                lambda_contrastive_loss = 0.1
                loss =  ce_loss + lambda_contrastive_loss * nce_loss_

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_ += loss.item()
                ce_loss_ += ce_loss.item()
                nce_loss_ += nce_loss_.item()
                batch_count += 1

            current_loss = loss_ / batch_count
            if current_loss < prev_loss:
                prev_loss = current_loss
                if (scheduler is not None) and (args.scheduler not in ["MULTILR", "ExponentialLR"]):
                    scheduler.step()
                    learning_rate = scheduler.get_last_lr()[0]
                    writer.add_scalar("LR", learning_rate, round)
                    round += 1
                break
            else:
                predict_model.load_state_dict(state_before_epoch)
                optimizer.load_state_dict(optimizer_state_before_epoch)
                retries += 1
                logger.info(f"Epoch {epoch+1} retry {retries} of {max_retries}, loss did not decrease.")
        else:
            logger.warning(f"Epoch {epoch+1} failed to decrease loss after {max_retries} retries.")
            prev_loss = current_loss

        if args.scheduler in ["MULTILR", "ExponentialLR"]:
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]
            writer.add_scalar("LR", learning_rate, epoch+1)

        eval_train = eval_model(predict_model, train_iter, device)
        eval_test = eval_model(predict_model, test_iter, device)

        """log"""
        logger.info("Epoch[%d/%d], loss:%.4f, CE loss:%.4f , info NCE loss:%.4f" 
                    % (epoch+1, num_epochs, loss_ / batch_count, ce_loss_ / batch_count, nce_loss_ / batch_count))
        logger.info("Epoch[%d/%d], train B-acc: %.4f(%.4f-%.4f), test B-acc: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[0][1], eval_train[0][0], eval_train[0][2],
                       eval_test[0][1], eval_test[0][0], eval_test[0][2]))
        logger.info("Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)"
                    % (epoch+1, num_epochs, eval_train[0][7], eval_train[0][6], eval_train[0][8], 
                       eval_test[0][7], eval_test[0][6], eval_test[0][8])) 
        
        """tensorboard"""
        writer.add_scalars("LOSS", {"ALL": loss_ / batch_count, "CE": ce_loss_ / batch_count,"info NCE": nce_loss_ / batch_count}, epoch + 1)        
        writer.add_scalars("ACC", {"B-ACC_train": eval_train[0][1], "B-ACC_test": eval_test[0][1]}, epoch + 1)
        writer.add_scalars("AUC", {"AUC_train": eval_train[0][7], "AUC_test": eval_test[0][7]}, epoch + 1)

        if (epoch+1) == num_epochs and args.savedisk:
            np.save(args.embed_path + "train_embed%d.npy" % (epoch + 1), eval_train[2])
            crr_train_record = eval_train[1]
            crr_train_erecord = eval_train[3]

            state0 = {
                "model": predict_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": loss_ / batch_count,
                "ce_loss": ce_loss_ / batch_count,
                "NUM_EPOCHS": num_epochs,
                "DEVICE": device
            }

            torch.save(state0, "{}/predict_model.pt".format(args.model_path))

            crr_train_record.to_csv("{}/crr_train_record_epoch_{:d}.csv".format(args.record_path, epoch + 1),
                                    header=True, index=False)
            crr_train_erecord.to_csv("{}/crr_train_erecord_epoch_{:d}.csv".format(args.record_path, epoch + 1),
                                    header=True, index=False)


def main(args, CLASSES, predict_model, model_architecture, time_tab):
    
    # set path for saving
    save_path = savePath(args.proj, time_tab)
    logger = get_logger(save_path.log_path)
    writer = SummaryWriter(save_path.writer_path)
    
    with open(save_path.argument_path, "w") as fw:
        json.dump(args.__dict__, fw, indent=2)
    desktop_path = r"C:\Users\azz\Desktop"  # 使用原始字符串避免转义问题
    check_dir(desktop_path)  # 确保目录存在
    
    # 将save_path.vis_path改为桌面路径
    save_path.vis_path = desktop_path


    PROJECT = args.proj
    batch_size = args.batch_size
    plr = args.plr
    elr = args.elr
    
    traindata = GenomicWSIDataset(PROJECT, "TRAIN", CLASSES)
    testdata = GenomicWSIDataset(PROJECT, "TEST", CLASSES)


    train_iter = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_iter = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    

    if args.optim == "ADAM":
        optimizer = torch.optim.Adam(predict_model.parameters(), lr=plr, weight_decay=0.01)
    elif args.optim == "RMSprop":
        optimizer = torch.optim.RMSprop(predict_model.parameters(), lr=plr, weight_decay=0.01)
    elif args.optim == "ADAMW":
        optimizer = torch.optim.AdamW(predict_model.parameters(), lr=plr, weight_decay=0.01)
    elif args.optim == "SGDNesterov":
        optimizer = torch.optim.SGD(predict_model.parameters(), lr=plr, nesterov=True, weight_decay=0.01, momentum=0.01)
    else:
        optimizer = torch.optim.SGD(predict_model.parameters(), lr=plr, weight_decay=0.01)
    

    if args.scheduler == "MULTILR":
        scheduler = MultiStepLR(optimizer, milestones=[int(s) for s in args.milestones.split("_")], gamma=args.lrgamma)
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    elif args.scheduler == "OneCycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=len(train_iter))
    elif args.scheduler == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=elr, max_lr=args.max_lr, mode="exp_range", step_size_up=50, gamma=args.lrgamma)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)
    elif args.scheduler == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, args.lrgamma)
    elif args.scheduler == "SequentialLR_1":
        sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
        sch2 = ConstantLR(optimizer, factor=0.1, total_iters=args.num_epochs)
        scheduler = SequentialLR(optimizer, [sch1, sch2], milestones=[int(args.milestones.split("_")[0])])
    elif args.scheduler == "SequentialLR_2":    
        sch1 = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.num_epochs, steps_per_epoch=len(train_iter))
        sch2 = ConstantLR(optimizer, factor=0.1, total_iters=args.num_epochs)
        scheduler = SequentialLR(optimizer, [sch1, sch2], milestones=[int(args.milestones.split("_")[0])])
    else:
        scheduler = None 
    

    logger.info("Running PROJECT: %s, P LEARNING RATE: %s, OPTIMIZER: %s, NUM EPOCHS: %d, BATCH SIZE: %d, DEVICES"
    ": %s, SMOOTHING: %s" 
                % (args.proj, str(args.plr), args.optim, args.num_epochs, args.batch_size, args.device, str(args.smoothing)))
    

    # record model architecture
    logger = logger_initial(logger, model_architecture)
    train_args = TrainingConfig(logger, writer, save_path, args)
    

    
    train(predict_model, train_iter, test_iter, CLASSES, optimizer, scheduler, train_args)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj", type=str, dest="proj", default="CRC-DX", help="POJRECT")
    parser.add_argument("--plr", type=float, dest="plr", default=0.002, help="LEARNING RATE FOR PREDICTED MODEL")
    parser.add_argument("--elr", type=float, dest="elr", default=0.002, help="LEARNING RATE FOR EMBED MODEL")
    parser.add_argument("--num_epochs", type=int, dest="num_epochs", default=200, help="NUM EPOCHS")
    parser.add_argument("--device", type=str, dest="device", default="cuda", help="DEVICE")
    parser.add_argument("--dropout1", type=float, dest="dropout1", default=0.8, help="DROPOUT")
    parser.add_argument("--dropout2", type=float, dest="dropout2", default=0.4, help="DROPOUT")
    parser.add_argument("--dropout3", type=float, dest="dropout3", default=0.4, help="DROPOUT")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="BATCH_SIZE")
    parser.add_argument("--optim", type=str, dest="optim", default="ADAMW", help="OPTIMIZER")
    parser.add_argument("--lam", type=float, dest="lam", default=1.0, help="Trade-OFF BETWEEN LOSS")
    parser.add_argument("--beta", type=float, dest="beta", default=1, help="Trade-OFF IN SIM LOSS")
    parser.add_argument("--mask_ratio", type=float, dest="mask_ratio", default=0.7, help="Percentage of masked gene features")
    parser.add_argument("--savedisk", type=bool, dest="savedisk", default=False, help="SAVE INTERMEDIATE OUTPUT")
    
    parser.add_argument("--milestone", type=str, dest="milestones", default="30_50", help="MILESTONES for MULTIPLE LR")
    parser.add_argument("--lrgamma", type=float, dest="lrgamma", default =0.1, help="DECAY RATE OF LR")
    
    parser.add_argument("--T_0", type=int, dest="T_0", default=10, help="T_0 FOR CosineAnnealingWarmRestarts")
    parser.add_argument("--T_mult", type=int, dest="T_mult", default=2, help="T_mult FOR CosineAnnealingWarmRestarts")

    parser.add_argument("--T_max", type=int, dest="T_max", default=10, help="T_max FOR CosineAnnealing")
    parser.add_argument("--max_lr", type=float, dest="max_lr", default=0.1, help="MAX LR FOR CyclicLR & OneCycleLR")
    parser.add_argument("--scheduler", type=str, dest="scheduler", default="None", help="ACTIVATE scheduler")
    parser.add_argument("--last_epoch", type=int, dest="last_epoch", default=10, help="LAST EPOCH FOR EVAL")

    parser.add_argument("--smoothing", type=float, dest="smoothing", default=0.03, help="ALPHA FOR LABEL SMOOTHING")
    parser.add_argument("--tau", type=float, dest="tau", default=0.02, help="TEMPERATURE FOR CONTRASTIVE LOSS")

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    # seed_everything(42)
    args = parse_args()
    args.device = "cuda"
    # args.lr = 1e-2
    #batch:训练数据集中随机选取的一部分数据，用于一次训练迭代
    args.batch_size = 2048
   

    #控制正则化项的强度，防止模型过拟合，出现过拟合时，需要增大该参数
    args.lam = 0.1
    
    args.beta = 0.5
    #args.savedisk = True
    args.savedisk = False

    CLASSES = ["MSS", "MSIMUT"]  
    class_num = len(CLASSES)

    #学习率调整参数
    args.lrgamma = 0.1

    run_time = 10

    time_tab = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("Running time tab:%s" % time_tab)

    # activate tensorboard
    command = "tensorboard --logdir=./TrainProcess/%s/%s/" % (args.proj, time_tab)
    process = subprocess.Popen(command)

    # set model architecture
    model_architecture = params_setup([args.dropout1, args.dropout2, args.dropout3])
    
    ##需要修改的地方
    # %% Replace the model for model design
    
    predict_model = MILAttenFusion(model_architecture, class_num)
  
    """ predict_model = MILAttenFusion(
        in_dim=model_architecture["dims"][0],
        hidden_dim=model_architecture["out_chans"][0],
        out_dim=model_architecture["out_chans"][1],
        dropout=model_architecture["dropout"][0],
        heads=model_architecture["heads"][0],
        class_num=class_num
    )"""
    main(args, CLASSES, predict_model, model_architecture, time_tab) 

    time.sleep(30)
    process.kill()