import torch
import torch.nn as nn

from collections import OrderedDict

from flatten import *

class LeNet(nn.Module):
    def __init__(self, droprate=0.5) -> None:
        super().__init__()
        """
        Architecture:
            - input
            - conv 
            - dropout
            - maxpool
            - conv
            - dropout
            - maxpool
            - flatten
            - dense
            - relu
            - dropout
            - out
        """
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2)),
            ('dropout1', nn.Dropout(p=droprate)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            ('conv2', nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2)),
            ('dropout2', nn.Dropout(p=droprate)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            ('flatten', Flatten()),
            ('dense3', nn.Linear(in_features=50*7*7, out_features=500)),
            ('relu3', nn.ReLU()),
            ('dropout3', nn.Dropout(p=droprate))
            
            ('final', nn.Linear(in_features=500, out_features=10))
        ]))
        
        def forward(self, x):
            return self.model(x)