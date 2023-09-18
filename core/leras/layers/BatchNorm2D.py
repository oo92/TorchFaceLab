import torch.nn as nn

class BatchNorm2D(nn.Module):
    """
    Custom BatchNorm2D layer using PyTorch.
    Currently not for training.
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None):
        super(BatchNorm2D, self).__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = torch.float32  # Use PyTorch's float32 as default
        self.dtype = dtype
        
        # Define weights
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=self.dtype))
        self.bias = nn.Parameter(torch.zeros(self.dim, dtype=self.dtype))
        # These are non-trainable parameters
        self.register_buffer('running_mean', torch.zeros(self.dim, dtype=self.dtype))
        self.register_buffer('running_var', torch.ones(self.dim, dtype=self.dtype))

    def forward(self, x):
        if x.shape[1] == self.dim:  # PyTorch uses NCHW by default
            shape = (1, self.dim, 1, 1)
        else:
            shape = (1, 1, 1, self.dim)  # This might not be needed, but keeping for consistency

        weight = self.weight.view(shape)
        bias = self.bias.view(shape)
        running_mean = self.running_mean.view(shape)
        running_var = self.running_var.view(shape)

        x = (x - running_mean) / torch.sqrt(running_var + self.eps)
        x *= weight
        x += bias
        return x

nn.BatchNorm2D = BatchNorm2D