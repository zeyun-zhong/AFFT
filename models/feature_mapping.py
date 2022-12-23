"""Implementation of different projection functions that map feature vectors with different sizes to a common size"""

import torch
from torch import nn as nn
from functools import partial
from torch.nn import functional as F


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, use_layernorm: bool = True):
        super().__init__()

        tmp = [nn.Linear(in_features, out_features), ContextGating(out_features)]

        if use_layernorm:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            tmp.append(norm_layer(out_features))

        layers = tmp  # deprecated: if in_features != out_features else [nn.Identity()]

        self.mapping = nn.Sequential(*layers)
        self.use_layernorm = use_layernorm

    def forward(self, x):
        return self.mapping(x)

    def __str__(self):
        return f'Gated linear mapping layer with use_layernorm: {self.use_layernorm}'


class Linear(nn.Module):
    """Implements the linear feature mapping layer"""
    def __init__(self, in_features, out_features, use_layernorm: bool = False, sparse_mapping=True):
        super().__init__()

        if sparse_mapping:
            layers = [nn.Linear(in_features, out_features, bias=False)
                      if in_features != out_features else nn.Identity()]
        else:
            layers = [nn.Linear(in_features, out_features, bias=False)]

        if use_layernorm:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            layers.append(norm_layer(out_features))

        self.mapping = nn.Sequential(*layers)
        self.use_layernorm = use_layernorm
        self.sparse_mapping = sparse_mapping

    def forward(self, x):
        return self.mapping(x)

    def __str__(self):
        return f'Linear mapping layer with use_layernorm: {self.use_layernorm}, ' \
               f'and sparse_mapping: {self.sparse_mapping}'


def get_activation_layer(name):
    act_layers = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'none': nn.Identity(),
    }
    assert name in act_layers.keys(), f'{name} is not supported in {list(act_layers.keys())}.'
    return act_layers[name]


class NonLinear(nn.Module):
    """Implements the non-linear feature mapping layer"""
    def __init__(self, in_features, out_features, use_layernorm: bool = False, activation='relu'):
        super().__init__()

        layers = [nn.Linear(in_features, out_features), get_activation_layer(activation)]

        if use_layernorm:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            layers.append(norm_layer(out_features))

        self.mapping = nn.Sequential(*layers)
        self.use_layernorm = use_layernorm
        self.activation = activation

    def forward(self, x):
        return self.mapping(x)

    def __str__(self):
        return f'Nonlinear mapping layer with use_layernorm: {self.use_layernorm}, ' \
               f'and activation: {self.activation}'
