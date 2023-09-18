import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Dense(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=True, use_wscale=False, maxout_ch=0, kernel_initializer=None, bias_initializer=None, trainable=True, dtype=None, **kwargs):
        super(Dense, self).__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        self.use_wscale = use_wscale
        self.maxout_ch = maxout_ch
        self.trainable = trainable

        # Compute the weight shape considering maxout
        if self.maxout_ch > 1:
            weight_shape = (in_ch, out_ch * maxout_ch)
        else:
            weight_shape = (in_ch, out_ch)

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(*weight_shape, dtype=dtype or torch.float32), requires_grad=trainable)
        
        # Weight scaling
        if self.use_wscale:
            gain = 1.0
            fan_in = np.prod(weight_shape[:-1])
            he_std = gain / np.sqrt(fan_in)
            self.wscale = torch.tensor(he_std, dtype=dtype or torch.float32)
            if kernel_initializer is None:
                init.normal_(self.weight, mean=0, std=1.0)
        else:
            if kernel_initializer is None:
                init.xavier_uniform_(self.weight)
            else:
                kernel_initializer(self.weight)
        
        # Initialize bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_ch, dtype=dtype or torch.float32), requires_grad=trainable)
            if bias_initializer is None:
                init.zeros_(self.bias)
            else:
                bias_initializer(self.bias)
                
    def forward(self, x):
        if self.use_wscale:
            x = torch.matmul(x, self.weight * self.wscale)
        else:
            x = torch.matmul(x, self.weight)

        if self.maxout_ch > 1:
            x = x.view(-1, self.out_ch, self.maxout_ch)
            x, _ = torch.max(x, dim=-1)

        if self.use_bias:
            x += self.bias.unsqueeze(0)

        return x

nn.Dense = Dense
