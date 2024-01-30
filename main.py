import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from Model import Deepnet
from sklearn.model_selection import KFold
import gc
from torch.utils.data import random_split
from index import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

def encode(DNA_sequence):
    torch_sq = []
    encode_ = {'A' : 0, 'C' : 1 , 'G' : 2 , 'T' : 3 }
    for base in DNA_sequence:
        base = encode_[base]
        torch_sq.append(base)
    x = torch.tensor(torch_sq)
    x = x.flatten()
    return x

def dataProcessing(path):
    file = open(path, "r")
    l1 = len(open(path).readlines())
    count = 0
    Training = [0] * l1
    for line in file:
        Data = line.strip('\n')
        Training[count] = encode(Data)
        count = count + 1
    return Training

def prepareData(PositiveCSV, NegativeCSV):
    Positive = dataProcessing(PositiveCSV)
    Negative = dataProcessing(NegativeCSV)
    
    len_data1 = len(Positive)
    len_data2 = len(Negative)
    
    Positive_y = torch.ones(len_data1, dtype=torch.float32)  
    Negative_y = torch.zeros(len_data2, dtype=torch.float32)
    
    for num in range(len(Positive)):
        Positive[num] = tuple((Positive[num],Positive_y[num]))
        Negative[num] = tuple((Negative[num],Negative_y[num]))
    Dataset = Positive + Negative
    return Dataset

def ModelTrainingWithCrossValidation(PositiveCSV, NegativeCSV, bs, net, lr, epochs, PATH, modelname, n_splits=5):
    """
    Trains a deep neural network model with K-Fold Cross-Validation.

    Args:
    - PositiveCSV: Path to the file containing positive samples.
    - NegativeCSV: Path to the file containing negative samples.
    - bs: Batch size.
    - net: Deep neural network model.
    - lr: Learning rate.
    - epochs: Number of epochs.
    - PATH: Path to save the model.
    - modelname: Name of the model.
    - n_splits: Number of splits for cross-validation (default: 5).

    """
    AllData = prepareData(PositiveCSV, NegativeCSV)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=520)

    for fold, (train_idx, test_idx) in enumerate(kf.split(AllData)):
        print(f"Starting fold {fold + 1}")
        net = Deepnet(channel=41, hidden=4, dropout=0.3, inputsize=4).to("cuda", non_blocking = True)
        # Split data
        train_dataset = [AllData[i] for i in train_idx]
        test_dataset = [AllData[i] for i in test_idx]

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True)
    
        criterion = nn.BCELoss()
        opt = torch.optim.Adadelta(net.parameters(),lr=lr,rho=0.9)
        highestAcc = None
        
        for epoch in range(epochs):
            train_pre, val_pre, train_labels, val_labels = [], [], [], []
            
            net = net.cuda()
            net.train()
            for num,(x,y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                opt.zero_grad(set_to_none=True)
                yhat = net.forward(x)
                yhat = yhat.flatten()
                loss = criterion(yhat,y)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 3, norm_type=2)
                opt.step()
                train_pre.extend(yhat.cpu().clone().detach().numpy().flatten().tolist())
                train_labels.extend(y.cpu().clone().detach().numpy().astype('int32').flatten().tolist()) 
            
            print("================================================", flush = True)
            print(f"epoch = {epoch+1}")
            for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](train_labels, train_pre, thresh = 0.5)
                    else:
                        metrics = metrics_dict[key](train_labels, train_pre)
                    print("train_" + key + ": " + str(metrics), flush=True) 
                    
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_pre, thresh = 0.5)
            print(f"{fold + 1} train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print(f"{fold + 1} train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print(f"{fold + 1} train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print(f"{fold + 1} train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)
            
            del x,y,yhat,tn_t, fp_t, fn_t, tp_t,train_labels,train_pre,metrics
            gc.collect()
            torch.cuda.empty_cache()
            
            print("------------------------------------------------", flush = True)
            
            net.eval()
            net = net.cuda()
            for num, (x, y) in enumerate(test_loader):
                with torch.no_grad():
                    x =  torch.LongTensor(x)
                    x = x.cuda()
                    y = y.cuda()
                    yhat = net(x)
                    yhat = yhat.flatten()
                    loss = criterion(yhat,y)
                    val_pre.extend(yhat.cpu().detach().numpy().flatten().tolist())
                    val_labels.extend(y.cpu().detach().numpy().astype('int32').flatten().tolist())
            loss_epoch = criterion(torch.tensor(val_pre).float(), torch.tensor(val_labels).float())
            
            print("validation loss:: "+ str(loss_epoch), flush = True)
            for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](val_labels, val_pre, thresh = 0.5)
                        
                        if(key == "f1"):
                            if (highestAcc == None) or (highestAcc < metrics):
                                highestAcc = metrics
                                torch.save(net.state_dict(), os.path.join(PATH, modelname + ".pt"))
                                print("Weights Saved")
                    
                    else:
                        metrics = metrics_dict[key](val_labels, val_pre)
                    print("validation_" + key + ": " + str(metrics), flush=True)
                    
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_pre, thresh = 0.5)
            print(f"{fold + 1} validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print(f"{fold + 1} validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print(f"{fold + 1} validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print(f"{fold + 1} validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)
            del x,y,yhat,tn_t, fp_t, fn_t, tp_t,val_labels,val_pre,metrics
            gc.collect()
            torch.cuda.empty_cache()
        

    
torch.manual_seed(520)
torch.cuda.manual_seed(520) 
PositivePath = ""
NegativePath = ""
net = Deepnet(feature=128,dropout=0.3,filter_num=128,seq_len=41).to("cuda",non_blocking=True)
Path = ""
modelname = ""

# net.load_state_dict (torch.load())
ModelTrainingWithCrossValidation(PositiveCSV=PositivePath, NegativeCSV=NegativePath,bs=128,net = net,lr=1.0,epoches=50,PATH=Path, modelname=modelname)
print("finish!")
