import torch.nn as nn

class ArchiBase(nn.Module):
    
    def __init__(self, *args, name=None, **kwargs):
        super(ArchiBase, self).__init__()
        self.name = name
        
    # Overridable 
    def flow(self, *args, **kwargs):
        raise Exception("This architecture does not support flow. Use model classes directly.")
    
    # Overridable
    def get_weights(self):
        pass

    
nn.ArchiBase = ArchiBase