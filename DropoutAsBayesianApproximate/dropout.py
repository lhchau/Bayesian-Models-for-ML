import torch
import torch.nn as nn
from torch.autograd import Variable

class MyDropout(nn.Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        
        self.p = p
        
        # multiplier is 1/(1-p)
        if self.p < 1:
            self.multiplier = 1 / (1 - p)
        else:
            self.multiplier = 1
    
    def forward(self, input):
        # if model.eval(), donot apply dropout
        if not self.training:
            return input
    
        # create random variable ~ Bernoulli(p) 
        selected = torch.Tensor(input.shape).uniform_(0, 1) > self.p
        
        if input.is_cuda():
            selected = Variable(selected.type(torch.cuda.FloatTensor),  requires_grad=False)
        else:
            selected = Variable(selected.type(torch.FloatTensor), requires_grad=False)
        
        # Multiply output by multiplier as paper, to normalize the weight
        return torch.mul(selected, input) * self.multiplier
