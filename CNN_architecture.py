import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)