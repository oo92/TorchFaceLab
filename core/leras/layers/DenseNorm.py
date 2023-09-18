import torch
import torch.nn as nn

class DenseNorm(nn.Module):
    def __init__(self, dense=False, eps=1e-06, dtype=None):
        super(DenseNorm, self).__init__()
        self.dense = dense
        
        if dtype is None:
            dtype = torch.float32
        self.eps = torch.tensor(eps, dtype=dtype)

    def forward(self, x):
        return x * (torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps).rsqrt()

nn.DenseNorm = DenseNorm