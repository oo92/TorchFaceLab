import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, in_ch, mlp_ch, kernel_initializer=None):
        super(AdaIN, self).__init__()

        self.in_ch = in_ch
        self.mlp_ch = mlp_ch

        if kernel_initializer is None:
            kernel_initializer = nn.init.kaiming_normal_

        # Define weight and bias using nn.Parameter so they get registered as learnable parameters automatically
        self.weight1 = nn.Parameter(torch.empty(mlp_ch, in_ch))
        self.bias1 = nn.Parameter(torch.zeros(in_ch))

        self.weight2 = nn.Parameter(torch.empty(mlp_ch, in_ch))
        self.bias2 = nn.Parameter(torch.zeros(in_ch))

        # Initialize weights using provided initializer
        kernel_initializer(self.weight1)
        kernel_initializer(self.weight2)

    def forward(self, x, mlp):
        gamma = torch.matmul(mlp, self.weight1) + self.bias1
        beta = torch.matmul(mlp, self.weight2) + self.bias2

        # Reshape gamma and beta for broadcasting
        gamma = gamma.view(-1, self.in_ch, 1, 1)
        beta = beta.view(-1, self.in_ch, 1, 1)

        # Calculate mean and standard deviation
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_std = x.std(dim=(2, 3), keepdim=True) + 1e-5

        # AdaIN transformation
        x = (x - x_mean) / x_std
        x *= gamma
        x += beta

        return x

nn.AdaIN = AdaIN