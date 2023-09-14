import numpy as np
import torch
from torch import nn

from .CA import CAInitializerSubprocessor

class Initializers(nn.Module):
    def __init__(self):
        super(Initializers, self).__init__()

    def forward(self, shape, dtype=None):
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def generate_batch(data_list, eps_std=0.05):
        # list of (shape, torch.dtype)
        return CAInitializerSubprocessor(data_list).run()
