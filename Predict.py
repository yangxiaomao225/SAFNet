import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt

import glob
from model.SAFNet import *

## Input Dataset
path = 'Input_data/*.npy'
Input_data_path = glob.glob(path) 
Input_data_path.sort()
SST_PATH = [i for i in Input_data_path if 'sst' in i]
ADT_PATH = [j for j in Input_data_path if 'adt' in j]

SSTs = []
for i in SST_PATH:
    SSTs.append(np.load(i))
SSTs_nd = np.array(SSTs) 

ADTs = []
for j in ADT_PATH:
    ADTs.append(np.load(j))
ADTs_nd = np.array(ADTs)

SSTs_tensor = torch.from_numpy(SSTs_nd).type(torch.float32).unsqueeze(1)
ADTs_tensor = torch.from_numpy(ADTs_nd).type(torch.float32).unsqueeze(1)

## Model loading
model = SAFNet()
model_para_path = 'SAFNet.pth'
model.load_state_dict(torch.load(model_para_path))

## Predicted Result 
z_pred = model(SSTs_tensor, ADTs_tensor)
predict=torch.where(z_pred>0.5,torch.ones_like(z_pred),torch.zeros_like(z_pred))
predict = predict.squeeze(1)
predict = predict.detach().numpy()

## Show the Result
list_name = ['20190101','20190401','20190701','20191001']
for i in range(4):    
    Y_line = predict[i]
    Y_line_1 = Y_line[::-1][27:115,0:60]
    plt.figure(figsize=(20,20))
    plt.imshow(Y_line_1, cmap = 'gray')
    plt.savefig(list_name[i]+'.png')
