import torch

"""
Input:
    - num_visible: number unit of visible
    - num_hidden: number unit of hidden
    - k: k-contrastive divergence (number samples for Gibbs sampling)
    - lr: learning rate
    - momentum_coeff: 
    - weight_decay: penalty hyperparameter
"""
class RBM():
    def __init__(self, num_visible, num_hidden, k, lr=1e-3, momentum_coeff=0.5, weight_decay=1e-4, use_cuda=True) -> None:
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.lr = lr
        self.momentum_coeff = momentum_coeff
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        
        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)
        
        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)
        
        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()
    
    # Gibbs sampling p(h|v)
    def sample_hidden(self, v):
        h = torch.matmul(v, self.weights) + self.hidden_bias
        h = torch.sigmoid(h)
        return h
    
    # Gibbs sampling p(v|h)
    def sample_visible(self, h):
        v = torch.matmul(h, self.weights.t()) + self.visible_bias
        v = torch.sigmoid(v)
        return v
    
    def contrastive_divergence(self, data_v):
        # Positive 
        
        # sample from data
        data_h = self.sample_hidden(data_v)
        data_h = (data_h >= self._random_probabilities(self.num_hidden)).float()
        data_ass = torch.matmul(data_v.t(), data_h)
        
        # Negative
        recon_h = data_h
        
        for step in range(self.k):
            recon_v = self.sample_visible(recon_h)
            recon_h = self.sample_hidden(recon_v)
            recon_h = (recon_h >= self._random_probabilities(self.num_hidden)).float()
        recon_ass = torch.matmul(recon_v.t(), recon_h)
        
        # Update parameters
        self.weights_momentum *= self.momentum_coeff
        self.weights_momentum += (data_ass - recon_ass)
        
        self.visible_bias_momentum *= self.momentum_coeff
        self.visible_bias_momentum += torch.sum(data_v - recon_v, dim=0)
        
        self.hidden_bias_momentum *= self.momentum_coeff
        self.hidden_bias_momentum += torch.sum(data_h - recon_h, dim=0)
        
        batch_size = data_v.shape[0]
        
        self.weights += self.weights_momentum * self.lr / batch_size
        self.visible_bias += self.visible_bias_momentum * self.lr / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.lr / batch_size
        
        # L2 regularization
        self.weights -= self.weights * self.weight_decay 
        
        # return error
        error = torch.sum((data_v - recon_v)**2)
        
        return error
    
    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)
        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities