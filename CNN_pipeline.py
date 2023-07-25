import pandas as pd
import numpy as np
import plotly
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from scipy.spatial.distance import cdist
from tqdm.notebook import trange, tqdm
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
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


max_x = 260
max_y = 277

interactions = pd.read_csv('all_imsis_cnn.csv')
label_dictionnary = pd.read_csv('CNN_trucks\labels_for_cnn.csv', index_col=False).set_index('hashed_imsi').to_dict()['label']

inputs = input_pipeline(interactions, label_dictionnary)
