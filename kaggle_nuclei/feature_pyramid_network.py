# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck, model_zoo, model_urls


class MaskMLP(nn.Module):
    def __init__(self, in_channels, out_channels, out_kernel=14):
        super().__init__()

        self.out_size = out_kernel
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, 512, 5, padding=2)
        self.conv2 = nn.ConvTranspose2d(512, out_channels, out_kernel)

    def forward(self, x):
        x = F.pad(x, (pad, pad))
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FPN(ResNet):
    def __init__(self, out_channels=2, pretrained=True):
        super().__init__(Bottleneck, [3, 4, 6, 3])

        d = 128
        self.out_channels = out_channels

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        # Top layer
        self.toplayer = nn.Conv2d(2048, d, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, d, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, d, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, d, kernel_size=1, stride=1, padding=0)

        self.mask_mlp = MaskMLP(d, out_channels)

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

    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        print(p2.shape, p3.shape, p4.shape, p5.shape)

        return p2, p3, p4, p5
