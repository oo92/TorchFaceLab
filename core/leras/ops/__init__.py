import numpy as np
import torch

# from core.leras import nn

from .CA import CAInitializerSubprocessor

class initializers():
    class ca:
        def __call__(self, shape, dtype=None):
            if dtype:
                return torch.zeros(shape, dtype=dtype)
            else:
                return torch.zeros(shape)

        @staticmethod
        def generate_batch(data_list, eps_std=0.05):
            # list of (shape, np.dtype)
            return CAInitializerSubprocessor(data_list).run()

# nn.initializers = initializers
