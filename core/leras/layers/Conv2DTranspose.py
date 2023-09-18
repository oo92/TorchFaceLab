import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Conv2DTranspose(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, strides=2, padding='SAME', use_bias=True, use_wscale=False, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs):
        super(Conv2DTranspose, self).__init__()
        
        if not isinstance(strides, int):
            raise ValueError("strides must be an int type")
        kernel_size = int(kernel_size)

        # Compute padding
        if padding == 'SAME':
            self.padding = kernel_size // 2
        elif padding == 'VALID':
            self.padding = 0
        else:
            raise ValueError("Unsupported padding type")

        self.use_wscale = use_wscale
        if self.use_wscale:
            gain = 1.0 if kernel_size == 1 else np.sqrt(2)
            fan_in = kernel_size * kernel_size * in_ch
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=dtype or torch.float32)
        
        # Create transposed convolution layer
        self.conv_transpose = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=strides, padding=self.padding, bias=use_bias)

        # Weight initialization
        if kernel_initializer is None:
            if self.use_wscale:
                init.normal_(self.conv_transpose.weight, mean=0, std=1.0)
            else:
                init.xavier_uniform_(self.conv_transpose.weight)
        else:
            kernel_initializer(self.conv_transpose.weight)

        # Bias initialization
        if use_bias:
            if bias_initializer is None:
                init.zeros_(self.conv_transpose.bias)
            else:
                bias_initializer(self.conv_transpose.bias)
                
    def forward(self, x):
        if self.use_wscale:
            return self.conv_transpose(x * self.wscale)
        else:
            return self.conv_transpose(x)

    def __str__(self):
        return f"{self.__class__.__name__} : in_ch:{self.conv_transpose.in_channels} out_ch:{self.conv_transpose.out_channels}"

nn.Conv2DTranspose = Conv2DTranspose