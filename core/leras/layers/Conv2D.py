import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, strides=1, padding='SAME', dilations=1, use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs):
        super(Conv2D, self).__init__()
        
        # Determine padding type
        if isinstance(padding, str):
            if padding == "SAME":
                self.padding = ( (kernel_size - 1) * dilations + 1 ) // 2
            elif padding == "VALID":
                self.padding = 0
            else:
                raise ValueError ("Wrong padding type. Should be VALID, SAME or INT or 4x INTs")
        else:
            self.padding = int(padding)

        # Use weight scaling or not
        self.use_wscale = use_wscale
        if self.use_wscale:
            gain = 1.0 if kernel_size == 1 else np.sqrt(2)
            fan_in = kernel_size * kernel_size * in_ch
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=dtype or torch.float32)

        # Create convolution layer
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=strides, padding=self.padding, dilation=dilations, bias=use_bias)

        # Weight initialization
        if kernel_initializer is None:
            if self.use_wscale:
                init.normal_(self.conv.weight, mean=0, std=1.0)
            else:
                init.xavier_uniform_(self.conv.weight)
        else:
            # Apply custom kernel initializer if provided
            kernel_initializer(self.conv.weight)

        # Bias initialization
        if use_bias:
            if bias_initializer is None:
                init.zeros_(self.conv.bias)
            else:
                # Apply custom bias initializer if provided
                bias_initializer(self.conv.bias)

    def forward(self, x):
        if self.use_wscale:
            return self.conv(x * self.wscale)
        else:
            return self.conv(x)

    def __str__(self):
        return f"{self.__class__.__name__} : in_ch:{self.conv.in_channels} out_ch:{self.conv.out_channels}"

nn.Conv2D = Conv2D