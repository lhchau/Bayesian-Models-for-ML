import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        """
        Input:
            - x: tensor [Batch_size, H, W, C]
        Output:
            - out: tensor [Batch_size, H*W*C]
        """
        x = x.view(x.size(0), -1)
        return x