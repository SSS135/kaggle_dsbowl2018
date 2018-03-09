# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet50, resnet101
import torch
from .conv_chunk import ConvChunk2d
from .skip_connections import DenseSequential, ResidualSequential


class MaskHead(nn.Module):
    conv_size = 4

    def __init__(self, in_channels, num_filters=128):
        super().__init__()
        assert FPN.mask_size % self.conv_size == 0
        self.in_channels = in_channels
        self.num_filters = num_filters

        self.preproc_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
        )
        self.mask_layers = []
        cur_size = self.conv_size
        cur_filters = num_filters
        while cur_size != FPN.mask_size:
            prev_filters = cur_filters
            cur_filters //= 2
            cur_size *= 2
            self.mask_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                DenseSequential(
                    nn.Sequential(
                        nn.Conv2d(prev_filters, cur_filters, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cur_filters),
                        nn.ReLU(True),
                    ),
                    nn.Sequential(
                        nn.Conv2d(prev_filters + cur_filters, cur_filters, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cur_filters),
                        nn.ReLU(True),
                    )
                ),
                nn.Conv2d(prev_filters + cur_filters * 2, cur_filters, 3, 1, 1),
            ))
        self.mask_layers.append(nn.Conv2d(cur_filters, 1, 3, 1, 1))
        self.mask_layers = nn.Sequential(*self.mask_layers)

    def forward(self, input):
        return self.preproc_layers(input)

    def predict_masks(self, x):
        assert x.shape == (x.shape[0], self.num_filters, self.conv_size, self.conv_size)
        return self.mask_layers(x)


class ScoreHead(nn.Module):
    def __init__(self, in_channels, num_scores, num_filters=128):
        super().__init__()
        assert FPN.mask_size % 4 == 0

        self.score_layers = nn.Sequential(
            DenseSequential(
                nn.Sequential(
                    nn.Conv2d(in_channels + num_filters * 0, num_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels + num_filters * 1, num_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels + num_filters * 2, num_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels + num_filters * 3, num_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                ),
            ),
            nn.Conv2d(in_channels + num_filters * 4, num_scores, 4),
        )

    def forward(self, input):
        score = self.score_layers(input)
        return score


class FPN(nn.Module):
    mask_size = 32
    mask_kernel_size = 4

    def __init__(self, num_scores=1, num_filters=256, num_head_filters=128):
        super().__init__()

        assert self.mask_size in (16, 32)
        assert self.mask_kernel_size == 4
        self.mask_pixel_sizes = (1, 2, 4, 8) if self.mask_size == 16 else (0.5, 1, 2, 4)
        self.mask_strides = (4, 8, 16, 32)
        self.resnet = resnet50(True)

        # Top layer
        self.toplayer = nn.Conv2d(2048, num_filters, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, num_filters, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, num_filters, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, num_filters, kernel_size=1, stride=1, padding=0)

        self.mask_head = MaskHead(num_filters, num_head_filters)
        self.score_head = ScoreHead(num_filters, num_scores, num_head_filters)

    def freeze_pretrained_layers(self, freeze):
        for p in self.resnet.parameters():
            p.requires_grad = not freeze

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        assert y.shape[2] % x.shape[2] == 0 and y.shape[3] % x.shape[3] == 0
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x, output_unpadding=0):
        c1 = self.resnet.conv1(x)
        c1 = self.resnet.bn1(c1)
        c1 = self.resnet.relu(c1)
        c1 = self.resnet.maxpool(c1)

        c2 = self.resnet.layer1(c1)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        if output_unpadding != 0:
            assert output_unpadding % 32 == 0
            ou = output_unpadding
            p5, p4, p3, p2 = [p[:, :, div:-div, div:-div] for (p, div) in
                              ((p5, ou // 32), (p4, ou // 16), (p3, ou // 8), (p2, ou // 4))]

        m5 = self.mask_head(p5), self.score_head(p5)
        m4 = self.mask_head(p4), self.score_head(p4)
        m3 = self.mask_head(p3), self.score_head(p3)
        m2 = self.mask_head(p2), self.score_head(p2)

        return m2, m3, m4, m5

    def predict_masks(self, x):
        return self.mask_head.predict_masks(x)
