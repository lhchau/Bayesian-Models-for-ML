import numpy as np
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable

from lenet import *

def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
      
def disable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.eval()

batch_size = 128
max_epoch = 300
lr = 0.1
momentum = 0.0
verbose = True
loss_ = []
test_error = []
test_accuracy = []

transform = transforms.Compose([transforms.ToTensor(), \
                                transforms.Normalize((0, 0, 0), (1, 1, 1))])

trainset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='data/', train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)

# Get test data
X_test, y_test = next(iter(testloader))
# If cuda is activated
X_test = X_test.cuda()

model = LeNet(droprate=0.0).cuda()

optimizer = optim.SGD(model.parameters, lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss().cuda()

for epoch in range(max_epoch):
    running_loss = 0
    for i, inputs, labels in enumerate(trainloader):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
    loss_.append(running_loss / len(trainloader))
    
    if verbose:
        print(f"Epoch {epoch+1} loss: {loss_[-1]}")
        
    ### Evaluate
    model.eval()
    outputs = model(Variable(X_test))
    _, y_test_pred = torch.max(outputs.data, 1) 
    model.train()
    ###
    
    test_accuracy.append(np.mean(y_test == y_test_pred))
    test_error.append(int(len(testset) * (1 - test_accuracy[-1])))
    if verbose:
        print(f"Test error: {test_error[-1]}; test accuracy: {test_accuracy[-1]}")

torch.save(model, f"mnist_lenet_overfit.pth")
# Prepare to save errors
test_error = list(map(str, test_error))

# Save test errors to plot figures
open("lenet_overfit_test_errors.txt", "w").write('\n'.join(','.join(test_error))) 