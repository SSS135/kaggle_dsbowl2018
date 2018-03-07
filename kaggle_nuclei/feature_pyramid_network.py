# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck, model_zoo, model_urls, resnet50, resnet101
import torch


class MaskMLP(nn.Module):
    def __init__(self, in_channels, f=256):
        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, f, 3, 1, 1),
            nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.Conv2d(f, f, 3, 1, 1),
            nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.Conv2d(f, f, FPN.mask_kernel_size),
            nn.Conv2d(f, FPN.mask_size * FPN.mask_size + 1, 1),
        )

    def forward(self, input):
        x = self.net(input)
        mask, score = x.split(FPN.mask_size * FPN.mask_size, 1)
        mask, score = mask, score.contiguous()
        mask = mask.permute(0, 2, 3, 1)
        mask = mask.contiguous().view(mask.shape[0], 1, *mask.shape[1:3], FPN.mask_size, FPN.mask_size)
        return mask, score


class FPN(nn.Module):
    mask_size = 16
    mask_kernel_size = 4

    def __init__(self, d=128):
        super().__init__()

        self.mask_pixel_sizes = (1, 2, 4, 8)
        self.mask_strides = (4, 8, 16, 32)
        self.resnet = resnet50(True)

        # Top layer
        self.toplayer = nn.Conv2d(2048, d, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, d, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, d, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, d, kernel_size=1, stride=1, padding=0)

        self.mask_mlp = MaskMLP(d)

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
        _, _, H, W = y.size()
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

        m5 = self.mask_mlp(p5)
        m4 = self.mask_mlp(p4)
        m3 = self.mask_mlp(p3)
        m2 = self.mask_mlp(p2)

        return m2, m3, m4, m5
