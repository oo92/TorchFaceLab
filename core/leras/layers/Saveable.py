import pickle, torch
import numpy as np
from pathlib import Path
from core import pathex
import torch.nn as nn

class Saveable(nn.Module):
    def __init__(self, name=None):
        super(Saveable, self).__init__()
        self.name = name

    def get_weights(self):
        # Return PyTorch tensors that should be initialized/loaded/saved
        return [param for param in self.parameters()]

    def get_weights_np(self):
        weights = self.get_weights()
        return [w.detach().numpy() for w in weights]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError('len of lists mismatch')

        for w, new_w in zip(weights, new_weights):
            w_shape = list(w.shape)
            if w_shape != list(new_w.shape):
                new_w = new_w.reshape(w_shape)
            w.data.copy_(torch.from_numpy(new_w))

    def save_weights(self, filename, force_dtype=None):
        d = {}
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        name = self.name

        for w in weights:
            w_val = w.detach().numpy().copy()
            w_name = w.name if hasattr(w, 'name') else str(id(w))
            w_name_split = w_name.split('/', 1)
            if name != w_name_split[0]:
                raise Exception("weight first name != Saveable.name")

            if force_dtype is not None:
                w_val = w_val.astype(force_dtype)

            d[w_name_split[1]] = w_val

        d_dumped = pickle.dumps(d, 4)
        pathex.write_bytes_safe(Path(filename), d_dumped)

    def load_weights(self, filename):
        filepath = Path(filename)
        if not filepath.exists():
            return False

        d_dumped = filepath.read_bytes()
        d = pickle.loads(d_dumped)

        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        try:
            for w in weights:
                w_name = w.name if hasattr(w, 'name') else str(id(w))
                w_name_split = w_name.split('/')
                if self.name != w_name_split[0]:
                    raise Exception("weight first name != Saveable.name")

                sub_w_name = "/".join(w_name_split[1:])
                w_val = d.get(sub_w_name, None)

                if w_val is None:
                    torch.nn.init.xavier_uniform_(w)
                else:
                    w_val = np.reshape(w_val, list(w.shape))
                    w.data.copy_(torch.from_numpy(w_val))
        except:
            return False

        return True

nn.Saveable = Saveable