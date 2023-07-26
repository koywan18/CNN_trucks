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
from CNN_architecture import *



def generate_dataloaders(interactions_file_path, label_file_path, bs = 8, max_x = 260, max_y = 277, test_size = 0.25):
    interactions = pd.read_csv(f'{interactions_file_path}')
    label_dictionnary = pd.read_csv(f'{label_file_path}', index_col=False).set_index('hashed_imsi').to_dict()['label']
    input_dataframe = input_pipeline(interactions, label_dictionnary, max_x = max_x, max_y = max_y)

    train_df, test_df = train_test_split(input_dataframe, test_size=test_size, random_state=15) #random state in fixed to have the same distribution every time
    x_train = np.array(train_df['array'].to_list())
    x_test = np.array(test_df['array'].to_list())
    y_train = train_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()
    # assuming `x_train` and `y_train` are your training data and labels as NumPy arrays
    train_out_of_batch = y_train.shape[0] % bs
    test_out_of_batch = y_test.shape[0] % bs

    x_train_tensor = torch.from_numpy(x_train)[:-train_out_of_batch,:,:]
    y_train_tensor = torch.from_numpy(y_train)[:-train_out_of_batch] #modify the train and test size in order to fit with the batch size
    x_test_tensor = torch.from_numpy(x_test) 
    x_test_tensor = x_test_tensor[:-test_out_of_batch,:,:]
    y_test_tensor = torch.from_numpy(y_test)
    y_test_tensor = y_test_tensor[:-test_out_of_batch]

    # create a TensorDataset from the training data and labels
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # create a DataLoader from the TensorDataset
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # create a DataLoader from the TensorDataset
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    return train_dataloader, test_dataloader

def run_training(model, train_dataloader, n_epochs, learning_rate, loss_fn):

    device = 'cuda' if torch.cuda else 'cpu'
    # your SGD optimizer
    optimizer_cl = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # and train for 10 epochs
    model =  model.to(device)
    l_t, a_t, tensor = train(model,train_dataloader,loss_fn,optimizer_cl,n_epochs = n_epochs)
    return model

def run_testing(model, test_dataloader, loss_fn):

    running_loss, predicted_y, y_test_bs, output_cumulated = test(model,test_dataloader, loss_fn)
    y_test_bs_np, output_cumulated = y_test_bs.to('cpu').numpy(), output_cumulated.to('cpu').numpy()
    return running_loss, predicted_y, y_test_bs_np, output_cumulated