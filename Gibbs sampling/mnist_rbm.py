import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from rbm import RBM


########## CONFIGURATION ##########
BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
EPOCHS = 20

DATA_FOLDER = './data'

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)
    
########## LOADING DATASET ##########
print('Loading dataset...')

mean = (0.1307,)
std = (0.3081,)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean, std)  # Normalize the tensor
])

train_dataset = datasets.MNIST(root=DATA_FOLDER, train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = datasets.MNIST(root=DATA_FOLDER, train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

########## TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        if CUDA:
            batch = batch.cuda()
            
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
    
########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
    if CUDA:
        batch = batch.cuda()
        
    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
    if CUDA:
        batch = batch.cuda()

    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()
    
########## CLASSIFICATION ##########
print('Classifying...')

clf = LogisticRegression(max_iter=200)
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

print('Result: %d/%d' % (sum(predictions == test_labels), len(test_labels)))