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

from CNN_trucks.CDR_preprocessing import *
from CNN_trucks.aggregated_features import *
from CNN_trucks.input_generation import *
from CNN_trucks.CNN_architecture import *


max_x = 260
max_y = 277

interactions = pd.read_csv('all_imsis_cnn.csv')
label_dictionnary = pd.read_csv('CNN_trucks\labels_for_cnn.csv', index_col=False).set_index('hashed_imsi').to_dict()['label']
input_dataframe = input_pipeline(interactions, label_dictionnary)

train_df, test_df = train_test_split(input_dataframe, test_size=0.31, random_state=15) #random state in fixed to have the same distribution every time
x_train = np.array(train_df['array'].to_list())
x_test = np.array(test_df['array'].to_list())
y_train = train_df['label'].to_numpy()
y_test = test_df['label'].to_numpy()
# assuming `x_train` and `y_train` are your training data and labels as NumPy arrays
train_out_of_batch = y_train.shape[0] % 8
test_out_of_batch = y_test.shape[0] % 8

x_train_tensor = torch.from_numpy(x_train)[:-train_out_of_batch,:,:]
y_train_tensor = torch.from_numpy(y_train)[:-train_out_of_batch] #modify the train and test size in order to fit with the batch size
x_test_tensor = torch.from_numpy(x_test) 
x_test_tensor = x_test_tensor[:-test_out_of_batch,:,:]
y_test_tensor = torch.from_numpy(y_test)
y_test_tensor = y_test_tensor[:-test_out_of_batch]

# create a TensorDataset from the training data and labels
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# create a DataLoader from the TensorDataset
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# create a DataLoader from the TensorDataset
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = 'cuda' if torch.cuda else 'cpu'

conv_class = classifier()
# choose the appropriate loss
#loss_fn = nn.NLLLoss()
loss_fn = nn.CrossEntropyLoss()

# your SGD optimizer
learning_rate = 1e-3
optimizer_cl = torch.optim.Adam(conv_class.parameters(),lr=learning_rate)
# and train for 10 epochs
conv_class =  conv_class.to(device)
l_t, a_t, tensor = train(conv_class,train_dataloader,loss_fn,optimizer_cl,n_epochs = 48)

tensor_test, prediction, y_test_bs, score_cumulated = test(conv_class,test_dataloader)
prediction_np, y_test_bs_np, score_np = prediction.to('cpu').numpy(), y_test_bs.to('cpu').numpy(), score_cumulated.to('cpu').numpy()
