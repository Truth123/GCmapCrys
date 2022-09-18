import numpy as np
import json
import yaml
from tqdm import tqdm
from Bio import SeqIO
import torch
from net.gat import GCmapCrys
import math
from sklearn.metrics import roc_auc_score


def savedict2Josn(item, save_path):
    item = json.dumps(item)
    
    try:
        with open(save_path, "w", encoding='utf-8') as f:
            f.write(item + ",\n")
    except Exception as e:
        print("write error==>", e)


def load_yaml(file):
    with open(file,'r',encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    return conf


def read_fasta(file):
    data = {}
    for it in SeqIO.parse(file, 'fasta'):
        name = str(it.id)
        value = str(it.seq)
        data[name] = value
    return data


def get_train_metrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]

    TP = FP = 0
    TN = np.sum(y_true==0)
    FN = np.sum(y_true==1)
    mcc = 0
    mcc_threshold = y_pred[0]+1
    confuse_matrix = (TP,FP,TN,FN)
    max_mcc = -1
    for index,score in enumerate(y_pred):
        if y_true[index] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        numerator = (TP*TN - FP*FN)
        denominator = (math.sqrt(TP+FP) * math.sqrt(TN+FN) * math.sqrt(TP+FN) * math.sqrt(TN+FP))
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator
        
        if mcc > max_mcc:
            max_mcc = mcc
            confuse_matrix = (TP, FP, TN, FN)
            mcc_threshold = score
    TP, FP, TN, FN = confuse_matrix
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)
    return mcc_threshold,TN,FN,FP,TP,Sen,Spe,Acc,max_mcc,AUC


def get_test_metrics(y_pred, y_true, threshold):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] >= threshold:
            TP += 1
        elif y_true[i] == 1 and y_pred[i] < threshold:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] >= threshold:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] < threshold:
            TN += 1
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)
    numerator = (TP*TN - FP*FN)
    denominator = (math.sqrt(TP+FP) * math.sqrt(TN+FN) * math.sqrt(TP+FN) * math.sqrt(TN+FP))
    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator
    return TN,FN,FP,TP,Sen,Spe,Acc,mcc,AUC
    


def evaluate(model, dataloader, device, is_train=True, threshold=0.5):
    model.eval()
    y_true = torch.tensor([],dtype=torch.int)
    y_score = torch.tensor([])
    for data in tqdm(dataloader):
        if isinstance(model, GCmapCrys):
            data = data.to(device)
            y = data.y
            out = model(data)
        else:
            x_feature, x_emb, y, id= data
            x_feature = x_feature.to(device)
            x_emb = x_emb.to(device)
            y = y.to(device)
            out = model(x_emb, x_feature)
        out = out.squeeze(dim=-1)
        out = torch.sigmoid(out)
        y_true = torch.cat((y_true, y.int().detach().cpu()))
        y_score = torch.cat((y_score, out.detach().cpu()))
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    if is_train:
        return get_train_metrics(y_score, y_true)
    else:
        return get_test_metrics(y_score, y_true, threshold)
