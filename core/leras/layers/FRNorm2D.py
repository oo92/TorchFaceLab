import torch
import torch.nn as nn

class FRNorm2D(nn.Module):
    """
    PyTorch implementation of
    Filter Response Normalization Layer: Eliminating Batch Dependence in theTraining of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None):
        super(FRNorm2D, self).__init__()
        self.in_ch = in_ch
        
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # In PyTorch, the preferred way to declare weights and biases is using nn.Parameter.
        self.weight = nn.Parameter(torch.ones(self.in_ch, dtype=self.dtype))
        self.bias = nn.Parameter(torch.zeros(self.in_ch, dtype=self.dtype))
        self.eps = nn.Parameter(torch.tensor(1e-6, dtype=self.dtype))

    def forward(self, x):
        # Assuming that the input tensor x has the shape (N, C, H, W), which is PyTorch's default format.

        # Compute nu2 as the mean square of x along spatial dimensions
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x_normalized = x * (1.0 / torch.sqrt(nu2 + self.eps.abs()))

        # Scale and shift the normalized tensor
        x_out = self.weight.view(1, self.in_ch, 1, 1) * x_normalized + self.bias.view(1, self.in_ch, 1, 1)

        return x_out

nn.FRNorm2D = FRNorm2D