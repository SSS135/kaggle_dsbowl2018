import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvChunk2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('weight', None)
        self.weight = torch.zeros(kernel_size ** 2 * channels, 1, kernel_size, kernel_size)
        w = torch.zeros(kernel_size ** 2)
        for i in range(kernel_size ** 2):
            w.zero_()[i] = 1
            self.weight[i * channels: (i + 1) * channels, 0] = \
                w.view(1, kernel_size, kernel_size).expand(channels, kernel_size, kernel_size)

    def forward(self, input):
        assert input.shape[1] == self.channels
        x = F.conv2d(input, Variable(self.weight), bias=None,
                     stride=self.stride, padding=self.padding, groups=self.channels)
        x = x.view(x.shape[0], self.kernel_size, self.kernel_size, self.channels, -1)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(-1, *x.shape[2:])
        return x