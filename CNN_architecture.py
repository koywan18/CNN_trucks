import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm.notebook import trange, tqdm
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms,datasets
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from CDR_preprocessing import *
from aggregated_features import *
from input_generation import *

device = 'cuda' if torch.cuda else 'cpu'

class classifier(nn.Module):
    
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=65, out_features=50)
        self.fc2 = nn.Linear(in_features=3460, out_features=2)
        self.counter = 0 # initialize counter

    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = F.max_pool2d(x1,4)
        x = torch.cat([torch.flatten(self.fc1(x1), start_dim = 1), x2], dim = 1)
        self.saved_tensor = x.detach().clone()
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def get_tensor(self):
        return self.saved_tensor


def train(model,data_loader,loss_fn,optimizer,n_epochs=1):
    model.train(True)
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    tensor = []
    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0

        for data in data_loader:
            x_train, y_train = data
            x_train = x_train.to(device)
            x_train = x_train.type(torch.float32)
            x1_train = x_train[:,:-1,:]
            x2_train = x_train[:,-1,:10]
            y_train = y_train.type(torch.LongTensor)
            y_train =  y_train.to(device)
            
            bs = x_train.size(0)

            outputs = model(x1_train, x2_train)
            loss = loss_fn(outputs,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,preds = torch.max(outputs,1)
            running_corrects += torch.sum(preds == y_train)
            running_loss += loss.data
            size += bs

            if epoch_num == n_epochs -1:
                tensor.append(model.get_tensor())
        epoch_loss = running_loss.item() / size
        epoch_acc = running_corrects.item() / size
        loss_train[epoch_num] = epoch_loss
        acc_train[epoch_num] = epoch_acc
        print('Train - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return loss_train, acc_train, tensor

def test(model,data_loader, loss_fn):
    bs = data_loader.batch_size
    y_test_final = torch.empty(bs*len(data_loader))
    output_cumulated = torch.empty((bs*len(data_loader),2))

    model.train(False)
    counter = 0
    running_corrects = 0.0
    running_loss = 0.0
    size = 0
    predicted_y = []
    for data in data_loader:
        x_test_bs, y_test_bs = data
        x_test_bs = x_test_bs.to(device)
        x_test_bs = x_test_bs.type(torch.float32)
        y_test_bs = y_test_bs.type(torch.LongTensor)
        x1_test = x_test_bs[:,:-1,:]
        x2_test = x_test_bs[:,-1,:10]
        y_test_bs =  y_test_bs.to(device)            
        outputs = model(x1_test, x2_test)
        loss = loss_fn(outputs,y_test_bs)
        _,preds = torch.max(outputs,1)

        running_corrects += torch.sum(preds == y_test_bs)
        running_loss += loss.data
        size += bs

        output_cumulated[counter:counter+bs] = outputs.detach()
        y_test_final[counter:counter+bs]= y_test_bs
        counter += bs
        predicted_y.append(model.get_tensor())
    print('Test - Loss: {:.4f} Acc: {:.4f}'.format(running_loss / size, running_corrects.item() / size))
    return running_loss, predicted_y, y_test_final, output_cumulated