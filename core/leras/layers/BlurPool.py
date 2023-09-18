import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BlurPool(nn.Module):
    def __init__(self, filt_size=3, stride=2):
        super(BlurPool, self).__init__()

        self.stride = stride
        pad = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.padding = pad + pad  # Double the pad list for 2D padding (height and width)

        if filt_size == 1:
            a = np.array([1.,])
        elif filt_size == 2:
            a = np.array([1., 1.])
        elif filt_size == 3:
            a = np.array([1., 2., 1.])
        elif filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a[:, None] * a[None, :]
        a = a / np.sum(a)
        a = a[:, :, None, None].astype(np.float32)  # Convert to float32 for torch tensor compatibility

        self.register_buffer('k', torch.tensor(a))

    def forward(self, x):
        k = self.k.expand(1, x.size(1), self.k.size(2), self.k.size(3))  # Copy the filter for each channel in the input
        x = F.pad(x, self.padding)
        x = F.conv2d(x, k, stride=self.stride, groups=x.size(1))  # Use depthwise convolution by setting groups to number of input channels

        return x

nn.BluePool = BlurPool