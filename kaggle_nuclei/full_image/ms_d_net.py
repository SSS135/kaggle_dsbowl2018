# sub-parts of the U-Net model

import torch
import torch.nn.functional as F
from torch import nn


class MSDLayer(nn.Module):
    def __init__(self, c_in, map_c, width, layer_depth, dilations):
        super().__init__()
        self.bn = nn.BatchNorm2d(width * map_c)
        self.convs = nn.ModuleList()
        for channel in range(width):
            dilation = dilations[(layer_depth * width + channel) % len(dilations)]
            conv = nn.Conv2d(c_in, map_c, 3, stride=1, padding=dilation, dilation=dilation)
            self.convs.append(conv)

    def forward(self, input):
        x = torch.cat([c(input) for c in self.convs], 1)
        x = self.bn(x)
        x = F.relu(x)
        x = torch.cat([input, x], 1)
        return x


class MSDNet(nn.Module):
    def __init__(self, in_channels, out_channels, map_channels, width, layers, dilations=(1, 2, 4, 8, 16), first_conv_channels=32):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, first_conv_channels, 5, stride=1, padding=2)
        self.layers = nn.ModuleList()
        cur_channels = first_conv_channels
        for layer_idx in range(layers):
            layer = MSDLayer(cur_channels, map_channels, width, layer_idx, dilations)
            self.layers.append(layer)
            cur_channels += map_channels * width
        self.combiner = nn.Conv2d(cur_channels, out_channels, 1)

    def forward(self, x):
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.combiner(x)
        return x