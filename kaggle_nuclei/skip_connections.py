import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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