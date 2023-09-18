import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TLU(nn.Module):
    """
    PyTorch implementation of
    Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None):
        super(TLU, self).__init__()
        
        self.in_ch = in_ch
        self.tau = nn.Parameter(torch.zeros(in_ch, dtype=dtype or torch.float32), requires_grad=True)
                
    def forward(self, x):
        tau = self.tau.view(1, self.in_ch, 1, 1)
        return torch.max(x, tau)

nn.TLU = TLU