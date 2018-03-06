import copy
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchsample.transforms as tst
import torchvision.transforms as tsf
from optfn.cosine_annealing import CosineAnnealingRestartLR
from optfn.param_groups_getter import get_param_groups
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import train_pad, train_size
from .iou import threshold_iou, iou
from .losses import dice_loss
from .transforms import RandomCrop, Pad
from .unet import UNet
from .dataset import resnet_norm_mean, resnet_norm_std
from .postprocessing_dataset import PostprocessingDataset
from optfn.gadam import GAdam
from .dataset import train_pad, train_size
from torch import nn

try:
    import ppo_pytorch.ppo_pytorch as rl
except:
    import ppo_pytorch as rl


class UNetActorCritic(rl.models.Actor):
    def __init__(self, obs_space, action_space, **kwargs):
        super().__init__(obs_space, action_space, **kwargs)
        self.unet = UNet(3, 3)

    def forward(self, input) -> rl.models.ActorOutput:
        bs = input.shape[0]
        input = input.view(bs, 3, train_size + train_pad * 2, train_size + train_pad * 2)
        x = self.unet(input)[:, :, train_pad:-train_pad, train_pad:-train_pad]
        values, mean, logstd = torch.unbind(x, 1)
        values = values * 0
        values = values.contiguous().view(bs, -1).mean(-1)
        probs = torch.cat([mean.contiguous().view(bs, -1), logstd.contiguous().view(bs, -1)], 1)
        return rl.models.ActorOutput(probs=probs, state_values=values)


class SimpleCNNActor(rl.models.Actor):
    def __init__(self, obs_space, action_space, **kwargs):
        super().__init__(obs_space, action_space, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 3, 7, stride=1, padding=3)

    def forward(self, input) -> rl.models.ActorOutput:
        bs = input.shape[0]
        input = input.view(bs, 3, train_size + train_pad * 2, train_size + train_pad * 2)

        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x[:, :, train_pad:-train_pad, train_pad:-train_pad]

        # x = x.contiguous().view(bs, 3, -1).mean(-1)
        # values = x[:, 0]
        # probs = x[:, 1:]

        values, mean, logstd = torch.unbind(x, 1)
        values = values * 0
        values = values.contiguous().view(bs, -1).mean(-1)
        probs = torch.cat([mean.contiguous().view(bs, -1), logstd.contiguous().view(bs, -1)], 1)
        return rl.models.ActorOutput(probs=probs, state_values=values)