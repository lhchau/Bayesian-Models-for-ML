import torch
import torch.nn as nn

class LaPlaceTrainer(object):
    def __init__(self, model, prior_precision=1, sigma_noise=1, last_layer=False):
        """
        LaPlace object that computes laplace approximation of map estimate

        Args:
            params (list) - the individual model parameters as a list
        """
        self.model = model
        self.last_layer = last_layer
        self.param_vector = self.params_to_vector()
        self.prior_precision = prior_precision
        self.sigma_noise = sigma_noise

    def jacobian(self, y, x, create_graph=False):
        """https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
        
        Args:
            y: 
            x:

        Returns:
            Jacobian of y with respect to x

        """
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        for i in range(len(flat_y)):
            grad_y[i] = 1.0
            (grad_x,) = torch.autograd.grad(
                flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
            )
            jac.append(grad_x.reshape(x.shape))
            grad_y[i] = 0.0
        return torch.stack(jac).reshape(y.shape + x.shape)

    def hessian(self, y, x):
        """https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
        
        Args:
            y:
            x:

        Returns:
            Hessian of y with respect to x
        
        """
        return self.jacobian(self.jacobian(y, x, create_graph=True), x)

    def params_to_vector(self):
        """
        returns a vector of all model parameters as a stacked vector
        model
        """
        if not self.last_layer:
            param_vector = torch.cat([param.view(-1) for param in self.model.parameters()])
        else:
            last_layer = list(self.model.children())[-1]
            param_vector = torch.cat([param.view(-1) for param in last_layer.parameters()])

        self.num_params = param_vector.shape[0]

        return param_vector

    def vector_to_params(self, param_vector):
        """
        returns the individual parameters from a vector
        
        Args:
            param_vector - given parameter vector to put into model
            
        """
        weight_idx = 0

        if not self.last_layer:
            param_iterator = self.model
        else:
            param_iterator = list(self.model.children())[-1] # last layer

        for param in param_iterator.parameters():
            param_len = param.numel()

            # updata parameter with param_vector slice
            param.data = param_vector[weight_idx: weight_idx+param_len].view_as(param).data

            weight_idx += param_len

    def criterion(self, pred, target, reduce=True):
        """MSE Loss
        
        Args:
            pred: model predictions
            target: ground truth values
            reudce: whether to reduce or return elementwise mse

        Returns:
            mse loss
        """
        if reduce:
            return ((target - pred) ** 2).sum()
        else:
            return (target - pred) ** 2

    def gradient(self, model):
        """Collects gradient of model output with respect to parameters.
        
        Args:
            model: model of which to gather derivates of parameters
        """
        grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
        return grad.detach()

    def jacobian_params(self, model, data, k=True):
        """Compute Jacobian of parameters.

        Args:
            model: model whose parameters to take gradient of
            data: input data to model

        Returns:
            Jacobian of model output w.r.t. to model parameters
        
        """
        model.zero_grad()
        output = model(data)
        Jacs = list()
        for i in range(output.shape[0]):
            rg = (i != (output.shape[0] - 1))
            output[i].backward(retain_graph=rg)
            jacs = self.gradient(model)
            model.zero_grad()
            Jacs.append(jacs)
        Jacs = torch.stack(Jacs)
        return Jacs.detach().squeeze(), output.detach()

    def last_layer_jacobian(self, model, X):
        """Compute Jacobian only of last layer

        Args:
            model: model of which to take the last layer
            X: model input

        Returns:
            Jacobian of model output w.r.t. to last layer parameters
        
        """
        model_no_last_layer = nn.Sequential(*list(model.children())[:-1])
        last_layer = list(model.children())[-1]
        input_to_last_layer = model_no_last_layer(X)
        jac, map = self.jacobian_params(last_layer, input_to_last_layer)
        return jac, map


    def compute_mean_and_cov(self, train_loader, criterion):
        """
        Compute mean and covariance for laplace approximation with general gauss newton matrix

        Args:
            train_loader: DataLoader
            criterion: Loss criterion used for training
        """
        precision = torch.eye(self.num_params) * self.prior_precision

        self.loss = 0
        self.n_data = len(train_loader.dataset)

        for X, y in train_loader:
            m_out = self.model(X)
            batch_loss = criterion(m_out, y)
            # jac is of shape N x num_params
            if not self.last_layer:
                jac, _ = self.jacobian_params(self.model, X)
            else:
                jac, _ = self.last_layer_jacobian(self.model, X)
                
            # hess is diagonal matrix of shape of NxN, where N is X.shape[0] or batch_size
            # hess = self.hessian(batch_loss, m_out).squeeze()
            hess = torch.eye(X.shape[0])
            precision += jac.T @ hess @ jac

            self.loss += batch_loss.item()

        self.n_data = len(train_loader.dataset)
        self.map_mean = self.params_to_vector()
        self.H = precision
        self.cov = torch.linalg.inv(precision)


    def linear_sampling(self, X, num_samples=100):
        """Prediction method with linearizing models and mc samples

        Args:
            X: prediction input
            num_samples: how many mc samples to draw

        Returns:
            laplace prediction and model map prediction
        
        """
        theta_map = self.params_to_vector()
        
        if not self.last_layer:
            jac, model_map = self.jacobian_params(self.model, X)
        else:

            jac, model_map = self.last_layer_jacobian(self.model, X)
            
        offset = model_map - jac @ theta_map.unsqueeze(-1)
        
        # reparameterization trick
        covs = self.cov @ torch.randn(len(theta_map), num_samples)

        theta_samples = theta_map + covs.T # num_samples x num_params
        preds = list()

        for i in range(num_samples):
            pred = offset + jac @ theta_samples[i].unsqueeze(-1)
            preds.append(pred.detach())

        preds = torch.stack(preds)

        return preds, model_map
        

    def predict(self, X, sampling_method="linear", num_samples=10000):
        """Prediction wrapper method to accustom different sampling methods
        
        Args:
            X: model input
            sampling_method: which sampling method to choose
            num_samples: how many mc samples to draw

        Returns:
            laplace prediction and model map prediction
        
        """
        if sampling_method == "linear":
            preds_la, preds_map = self.linear_sampling(X, num_samples)

        return preds_la, preds_map