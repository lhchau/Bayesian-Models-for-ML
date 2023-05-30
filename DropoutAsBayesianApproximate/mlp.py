import torch
import torch.nn as nn

from collections import OrderedDict

from dropout import *

class MLP(nn.Module):
    def __init__(self, hidden_layers=[800, 800], droprates=[0, 0]) -> None:
        super().__init__()
        
        self.model = nn.Sequential()
        self.model.add_module("dropout0", MyDropout(p=droprates[0]))
        self.model.add_module("input", nn.Linear(in_features=28*28, out_features=hidden_layers[0]))
        self.model.add_module("tanh", nn.Tanh())
        
        # Add hidden layers
        for i,d in enumerate(hidden_layers[:-1]):
            self.model.add_module(f"dropout_hidden{i+1}", MyDropout(p=droprates[1]))
            self.model.add_module(f"hidden{i+1}", nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.model.add_module(f"tanh_hidden{i+1}", nn.Tanh())
        self.model.add_module("final", nn.Linear(hidden_layers[-1], 10))    
        
    def forward(self, x):
        # Flatten to [Batchsize, H*W*C]
        x = x.view(x.shape[0], 28*28) 
        x = self.model(x)
        return x   