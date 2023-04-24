import torch
import torch.nn as nn

def sinusoid_data(n=150, sigma_noise=0.3):
    torch.manual_seed(42)
    # create simple sinusoid data set
    x_train = (torch.rand(n) * 10).unsqueeze(-1)
    y_train = torch.sin(x_train) + torch.randn_like(x_train) * sigma_noise
    x_test = torch.linspace(-10, 13, 500).unsqueeze(-1)
    y_test = torch.sin(x_test) + torch.randn_like(x_test) * sigma_noise

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

def criterion(pred, target, reduce=True):
    """ MSE Loss
    """
    if reduce:
        return ((target - pred)**2).mean()
    else:
        return (target - pred)**2
    
def construct_model(num_features, n_hidden_units=10):
    model = nn.Sequential(
        nn.Linear(num_features, n_hidden_units),
        nn.Tanh(),
        nn.Linear(n_hidden_units, 1)
    )
    
    return model

def train(X, y, model, optimizer, n_epochs):
    ##################
    ## MAP Training ##
    ##################
    for i in range(n_epochs):
        pred = model(X)
        nll = criterion(pred, y)
        print(nll)
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()