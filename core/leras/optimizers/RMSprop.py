import numpy as np
import torch
from torch import nn as t_nn

class RMSprop(t_nn.Module):
    def __init__(self, trainable_params, lr=0.001, rho=0.9, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None):
        super(RMSprop, self).__init__()

        if name is None:
            raise ValueError('name must be defined.')

        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.lr = lr
        self.rho = rho
        self.clipnorm = clipnorm

        self.iterations = torch.tensor(0, dtype=torch.int64)
        self.accumulators_dict = {}
        self.lr_rnds_dict = {}
        
        for name, param in trainable_params:
            self.accumulators_dict[name] = torch.zeros_like(param.data)
            if self.lr_dropout != 1.0:
                self.lr_rnds_dict[name] = torch.bernoulli(torch.ones_like(param.data) * self.lr_dropout)

    def step(self, grads_vars):
        updates = []

        if self.clipnorm > 0.0:
            norm = torch.sqrt(sum([(g ** 2).sum() for g, v in grads_vars]))
        
        self.iterations += 1
        
        for i, (name, g, v) in enumerate(grads_vars):
            if self.clipnorm > 0.0:
                g = torch.nn.utils.clip_by_norm(g, self.clipnorm, norm_type=2) * (norm if norm.is_nonzero() else 1.0)

            a = self.accumulators_dict[name]

            new_a = self.rho * a + (1. - self.rho) * g ** 2

            lr = torch.tensor(self.lr, dtype=g.dtype)
            if self.lr_cos != 0:
                lr *= (torch.cos(self.iterations.to(dtype=g.dtype) * (2*np.pi/ float(self.lr_cos))) + 1.0) / 2.0

            v_diff = - lr * g / (torch.sqrt(new_a) + np.finfo(g.dtype).eps)
            
            if self.lr_dropout != 1.0:
                lr_rnd = self.lr_rnds_dict[name]
                v_diff *= lr_rnd
            v.data += v_diff

            self.accumulators_dict[name] = new_a
