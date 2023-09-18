import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv2D(nn.Module):
    def __init__(self, in_ch, kernel_size, strides=1, padding='SAME', depth_multiplier=1, 
                 dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, 
                 bias_initializer=None, trainable=True, dtype=None):
        
        super(DepthwiseConv2D, self).__init__()
        
        if dtype is None:
            dtype = torch.float32
        
        if padding == "SAME":
            padding = ( (kernel_size - 1) * dilations + 1 ) // 2
        elif padding == "VALID":
            padding = 0
        else:
            raise ValueError("Wrong padding type. Should be VALID, SAME, or int.")

        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.in_ch = in_ch
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        if self.use_wscale:
            gain = 1.0 if self.kernel_size == 1 else np.sqrt(2)
            fan_in = self.kernel_size * self.kernel_size * self.in_ch
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=dtype)

        self.weight = nn.Parameter(torch.empty(self.kernel_size, self.kernel_size, self.in_ch, self.depth_multiplier, dtype=dtype))
        if self.kernel_initializer:
            nn.init.xavier_uniform_(self.weight)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.in_ch * self.depth_multiplier, dtype=dtype))
            if self.bias_initializer:
                nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight = self.weight
        if self.use_wscale:
            weight = weight * self.wscale
        
        x = F.conv2d(x, weight, self.bias, stride=self.strides, padding=self.padding, dilation=self.dilations, groups=self.in_ch)
        return x

    def __str__(self):
        return f"{self.__class__.__name__} : in_ch:{self.in_ch} depth_multiplier:{self.depth_multiplier}"

nn.DepthwiseConv2D = DepthwiseConv2D