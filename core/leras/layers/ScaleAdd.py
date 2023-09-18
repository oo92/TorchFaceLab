import torch
import torch.nn as nn

class ScaleAdd(nn.Module):
    def __init__(self, ch, dtype=None):
        super(ScaleAdd, self).__init__()
        
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.ch = ch
        
        # Initialize weights
        self.weight = nn.Parameter(torch.zeros(self.ch, dtype=self.dtype))
        
    def forward(self, inputs):
        # Depending on the data format, choose the shape for reshaping the weights
        # Here, I'm assuming a default of "NCHW" for PyTorch. If you need "NHWC", you'll have to adjust.
        shape = (1, self.ch, 1, 1)
        
        # Reshape the weights
        weight = self.weight.view(shape)
        
        # Extract inputs
        x0, x1 = inputs
        
        # Compute the output
        x = x0 + x1 * weight
        
        return x
    
nn.ScaleAdd = ScaleAdd