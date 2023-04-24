from utils import *
from hessian import *
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# from laplace import Laplace

data = sinusoid_data()
x_train, y_train, x_test, _ = (
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
    )
model = construct_model(num_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x_scaler = StandardScaler()
x_train = torch.from_numpy(x_scaler.fit_transform(x_train))
x_train = x_train.float()

### map training ###
n_epochs = 1000
train(x_train, y_train, model, optimizer, n_epochs=n_epochs)

##################
## Fit LaPlace  ##
##################
last_layer = False
LA = LaPlaceTrainer(model=model, last_layer=last_layer)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=150)
LA.compute_mean_and_cov(
    train_loader=train_loader, criterion=criterion
)