import torch
import torch.nn as nn

class InstanceNorm2D(nn.Module):
    def __init__(self, in_ch, dtype=None):
        super(InstanceNorm2D, self).__init__()
        self.in_ch = in_ch
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        
        # Instead of using tf.get_variable(), we can use nn.Parameter to declare weights
        # Note that in PyTorch, the preferred way to declare weights and biases is using nn.Parameter.
        self.weight = nn.Parameter(torch.randn(self.in_ch, dtype=self.dtype))
        self.bias = nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))
        
        # Initialize the weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Assuming that data_format "NHWC" corresponds to shape (N, H, W, C) in TensorFlow,
        # this is equivalent to (N, C, H, W) in PyTorch, which is PyTorch's default format.
        
        # Computing mean and std dev along the spatial dimensions
        x_mean = x.mean(dim=[2,3], keepdim=True)
        x_std = x.std(dim=[2,3], keepdim=True) + 1e-5
        
        # Perform instance normalization
        x_normalized = (x - x_mean) / x_std
        x_out = self.weight.view(1, self.in_ch, 1, 1) * x_normalized + self.bias.view(1, self.in_ch, 1, 1)
        
        return x_out

nn.InstanceNorm2D = InstanceNorm2D