import torch
import torch.nn as nn


class DenseSequential(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = torch.cat([input, module(input)], 1)
        return input


class ResidualSequential(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = input + module(input)
        return input