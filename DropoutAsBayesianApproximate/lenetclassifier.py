import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lenet import *

class LeNetClassifier(nn.Module):
    def __init__(self, droprate=0.5, batch_size=128, max_epoch=300, \
                lr=0.1, momentum=0) -> None:
        super().__init__()
        """
        Input: 
            - hidden_layers: in_features at each hidden layer
            - droprates: rate p(dropout) at each hidden layer
            - batch_size: number images in each iteration
            - max_epoch: number epochs
            - lr: learning rate 
            - momentum: lr + momentum to initialize optimizer
        """
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model = LeNet(droprate=droprate)
        self.model.cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.loss_ = []
        self.test_accuracy = []
        self.test_error = []

    def fit(self, trainset, testset, verbose=True):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
        
        # Get test data
        X_test, y_test = next(iter(testloader))
        # If cuda is activated
        X_test = X_test.cuda()
        for epoch in range(self.max_epoch):
            running_loss = 0
            for i, inputs, labels in enumerate(trainloader):
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
            self.loss_.append(running_loss / len(trainloader))
            
            if verbose:
                print(f"Epoch {epoch+1} loss: {self.loss_[-1]}")
            y_test_pred = self.predict(X_test).cpu()
            self.test_accuracy.append(np.mean(y_test == y_test_pred))
            self.test_error.append(int(len(testset) * (1 - self.test_accuracy[-1])))
            if verbose:
                print(f"Test error: {self.test_error[-1]}; test accuracy: {self.test_accuracy[-1]}")
        return self
        
    def predict(self, x):
        model = self.model.eval()
        outputs = model(Variable(x))
        _, pred = torch.max(outputs.data, 1)
        model = self.model.train()
        return pred
    
    def __str__(self) -> str:
        return f"Layers: {self.model}; \ndropout rates: {self.droprates}"