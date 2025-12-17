# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:39:19 2020

@author: Administrator
"""

import numpy as np
#from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score
import torch
import pandas as pd
import time 
#计算模型准度评价参数


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_tsne(embeddings, patient_ids, vis_path, epoch):
    # 提取特征和患者ID（假设最后一列是标签，患者ID单独传递）
    features = embeddings[:, :-1]  # 特征部分
    labels = embeddings[:, -1]     # 标签部分
    
    # 降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)
    
    # 创建调色板（基于患者ID的哈希值）
    unique_patients = list(set(patient_ids))
    palette = sns.color_palette("husl", n_colors=len(unique_patients))
    patient_to_color = {pid: palette[i] for i, pid in enumerate(unique_patients)}
    
    # 绘图
    plt.figure(figsize=(12, 8))
    for pid in unique_patients:
        idx = np.where(patient_ids == pid)[0]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], 
                    color=patient_to_color[pid], 
                    label=f'Patient {pid}',
                    alpha=0.6)
    
  
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{vis_path}/tsne_epoch_{epoch}.png", dpi=300)
    
    plt.grid(False)
    plt.close()
    
def save_record(patchID, y_true, y_probs):
    record_dict = dict()
    record_dict["patchID"] = patchID
    record_dict["y_true"] = y_true
    for i in range(y_probs.shape[1]):
        record_dict["y_probs%d" % i] = y_probs[:, i]
    record_df = pd.DataFrame.from_dict(record_dict)

    return record_df

def ensemble_patient(patientID, y_true, y_probs):
    # patientID: a list with patient id
    # y_probs: prediction score of a sample sample_num x class_num
    # y_true: ground-truth label of a sample
    uq_patientID = list(set(patientID))

    ensemble_y_probs = []
    ensemble_y_true = []

    for uq_id in uq_patientID:
        bool_idx = [True if uq_id == id else False for id in patientID]
        ensemble_y_probs.append(y_probs[bool_idx, :].mean(axis=0))
        ensemble_y_true.append(y_true[bool_idx].mean())

    ensemble_y_true = np.array(ensemble_y_true).ravel()
    ensemble_y_probs = np.array(ensemble_y_probs)

    return uq_patientID, ensemble_y_probs, ensemble_y_true


def auc(target, score):
    if len(set(target)) > 2:
        auc_ = roc_auc_score(target, score, average="macro", multi_class="ovr")
    else:
        auc_ = roc_auc_score(target, score[:, 1])
    return auc_


def get_prec_sen_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    return precision, sensitivity, specificity


def full_evaluation(y_true, y_probs):
    # y_pred = np.argmax(y_probs, axis=1)
    # acc = np.sum(y_pred ==y_true) / len(y_true)
    # f1_score_ = f1_score(y_true, y_pred, average="weighted")
    # # auc_ = auc(y_true, y_probs)
    # low_auc, mid_auc, up_auc = bootstrap(y_true, y_probs, 1000, 0.95)
    # auc_ = (low_auc, mid_auc, up_auc)
    # return acc, f1_score_, auc_

    result = bootstrap_ap(y_true, y_probs, 1000, 0.95)
    return result


def eval_model(model, data, device):
    y_true = []
    y_probs = []
    patientID = []
    patchID = []
    embeddings = []
    model.eval()
    # timeflag1 = time.time()
    with torch.no_grad():
        for wdata_, y, patientID_, patchID_ in data:
            wdata_ = wdata_.float().to(device)
            embedding_, linear_prob = model(wdata_)
            prob = torch.softmax(linear_prob, dim=1)
            y_probs.append(prob)
            embeddings.append(embedding_)
            y_true.append(y)
            patientID += patientID_
            patchID += patchID_

    # timeflag2 = time.time()
    
    # print("time lag: %.4f" % (timeflag2 - timeflag1))

    # timeflag3 = time.time()
    
    y_true = torch.cat(y_true, dim=0).numpy().ravel()
    y_probs = torch.cat(y_probs, dim=0).cpu().numpy()
    record_df = save_record(patchID, y_true, y_probs)
    ensemble_patientID, ensemble_y_probs, ensemble_y_true = ensemble_patient(patientID, y_true, y_probs)
    # acc, f1_score_, auc_ = full_evaluation(ensemble_y_true, ensemble_y_probs)
    metric = full_evaluation(ensemble_y_true, ensemble_y_probs)

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    embeddingsy = np.concatenate((embeddings, y_true.reshape(-1, 1)), axis=1)

    # timeflag4 = time.time()
    # print("time lag: %.4f" % (timeflag4 - timeflag3))

    ensemble_record = {"patientID": ensemble_patientID, "y_true": ensemble_y_true}
    for i in range(ensemble_y_probs.shape[1]):
        ensemble_record["y_probs%d" % i] = ensemble_y_probs[:, i]
    
    ensemble_record_df = pd.DataFrame.from_dict(ensemble_record)
    

    return metric, record_df, embeddingsy, ensemble_record_df


def bootstrap(target, score, B, c):
   
    n = len(target)
    sample_result_arr = []
    count = 0
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        target_sample = target[index_arr]
        score_sample = score[index_arr,:]
        if len(set(target_sample)) == 1:
            continue

        sample_result = auc(target_sample,score_sample)
        sample_result_arr.append(sample_result)
        count += 1

    a = 1 - c
    k1 = int(count * a / 2)
    k2 = int(count * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]
    mid = auc_sample_arr_sorted[int(count/2)]

    return lower, mid, higher


def bootstrap_ap(target, score, B, c):
    
    n = len(target)
    sample_result_arr_acc = []
    #sample_result_arr_f1 = []
    sample_result_arr_auc = []
    count = 0
    while True:
        index_arr = np.random.randint(0, n, size=n)
        target_sample = target[index_arr]
        if len(set(target_sample)) == 1:
            continue
        
        score_sample = score[index_arr]
        y_pred = np.argmax(score_sample, axis=1)
        
        # sample_result_acc = np.sum(y_pred == target_sample) / len(target_sample)
        sample_result_acc = balanced_accuracy_score(target_sample, y_pred)
        #sample_result_f1 = f1_score(target_sample, y_pred, average="weighted")
        sample_result_auc = auc(target_sample, score_sample)
        
        sample_result_arr_acc.append(sample_result_acc)
        #sample_result_arr_f1.append(sample_result_f1)
        sample_result_arr_auc.append(sample_result_auc)
        
        count += 1
        if count > B:
            break


    a = 1 - c
    k1 = int(count * a / 2)
    k2 = int(count * (1 - a / 2))
    ap_sample_arr_auc_sorted = sorted(sample_result_arr_auc)
    auc_lower = ap_sample_arr_auc_sorted[k1]
    auc_higher = ap_sample_arr_auc_sorted[k2]
    auc_mid = ap_sample_arr_auc_sorted[int(count/2)]


    ap_sample_arr_acc_sorted = sorted(sample_result_arr_acc)
    acc_lower = ap_sample_arr_acc_sorted[k1]
    acc_higher = ap_sample_arr_acc_sorted[k2]
    acc_mid = ap_sample_arr_acc_sorted[int(count/2)]

    return tuple([acc_lower, acc_mid, acc_higher,None,None,None, auc_lower, auc_mid, auc_higher])
    #return tuple([acc_lower, acc_mid, acc_higher, auc_lower, auc_mid, auc_higher])

