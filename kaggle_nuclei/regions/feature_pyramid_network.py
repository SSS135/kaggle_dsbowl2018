# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import itertools
import math

import torch
import torch.nn.functional as F
from optfn.se_module import SELayer
from optfn.shuffle_conv import ShuffleConv2d
from pretrainedmodels import dpn92, dpn68, dpn68b, nasnetalarge, resnext101_32x4d
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.instancenorm import _InstanceNorm
from torchvision.models import resnet50

from ..utils import unpad
from ..lexp import lexp
from ..settings import box_padding


class MaskHead(nn.Module):
    def __init__(self, num_filters, region_size, mask_size):
        super().__init__()
        assert mask_size % region_size == 0
        self.num_filters = num_filters
        self.region_size = region_size
        self.mask_size = mask_size

        self.conv_mask_layers = [
            BatchChannels(num_filters),
            ResBlock(num_filters),
            AdaptiveFeaturePooling(FPN.num_feature_groups),

            ShuffleConv2d(num_filters, num_filters, 3, 1, 1, bias=False, groups=4),
            nn.InstanceNorm2d(num_filters, affine=True),
            nn.ReLU(True),
        ]
        cur_size = region_size
        cur_filters = num_filters
        while cur_size != mask_size:
            prev_filters = cur_filters
            cur_filters //= 2
            cur_size *= 2
            self.conv_mask_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ShuffleConv2d(prev_filters, cur_filters, 3, 1, 1, bias=False, groups=4),
                nn.InstanceNorm2d(cur_filters, affine=True),
                nn.ReLU(True),
            ))
        self.conv_mask_layers.append(nn.Sequential(
            nn.Conv2d(cur_filters, 1, 1),
        ))
        self.conv_mask_layers = nn.Sequential(*self.conv_mask_layers)

    def forward(self, input, unpadding):
        return unpad(input, unpadding).contiguous()

    def predict_masks(self, x):
        return self.conv_mask_layers(x.contiguous())


class ScoreHead(nn.Module):
    def __init__(self, num_filters, num_scores):
        super().__init__()
        self.score_layers = nn.Sequential(
            BatchChannels(num_filters),
            ResBlock(num_filters),
            AdaptiveFeaturePooling(FPN.num_feature_groups),

            ShuffleConv2d(num_filters, num_filters, 3, 1, 1, bias=False, groups=4),
            nn.InstanceNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            ShuffleConv2d(num_filters, num_filters, 3, 1, 1, bias=False, groups=4),
            nn.InstanceNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, num_scores, 1),
        )
        init_fg_conf = 0.1
        self.score_layers[-1].bias.data.fill_(-math.log((1 - init_fg_conf) / init_fg_conf))

    def forward(self, input, unpadding):
        score = unpad(self.score_layers(input), unpadding).contiguous()
        return score


class BoxHead(nn.Module):
    def __init__(self, num_filters, region_size):
        super().__init__()
        self.region_size = region_size
        self.layers = nn.Sequential(
            BatchChannels(num_filters),
            ResBlock(num_filters),
            AdaptiveFeaturePooling(FPN.num_feature_groups),

            ShuffleConv2d(num_filters, num_filters, 3, 1, 1, bias=False, groups=4),
            nn.InstanceNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            ShuffleConv2d(num_filters, num_filters, 3, 1, 1, bias=False, groups=4),
            nn.InstanceNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, 6, 1),
        )
        self.layers[-1].bias.data.fill_(0)
        # self.layers[-1].weight.data.mul_(0.2)

    def forward(self, input, unpadding):
        raw_boxes = unpad(self.layers(input), unpadding).contiguous()
        ih, iw = raw_boxes.shape[2:]
        anchor_pos_y = torch.arange(ih).type_as(raw_boxes.data).view(1, -1, 1).add_(0.5)
        anchor_pos_x = torch.arange(iw).type_as(raw_boxes.data).view(1, 1, -1).add_(0.5)
        anchor_pos_y, anchor_pos_x = Variable(anchor_pos_y), Variable(anchor_pos_x)
        b_y = raw_boxes[:, 0] * self.region_size + anchor_pos_y
        b_x = raw_boxes[:, 1] * self.region_size + anchor_pos_x
        b_h = lexp(raw_boxes[:, 2]) * self.region_size * (1 + box_padding)
        b_w = lexp(raw_boxes[:, 3]) * self.region_size * (1 + box_padding)
        b_dir = raw_boxes[:, 4:6] / raw_boxes[:, 4:6].pow(2).sum(1, keepdim=True).add(1e-6).sqrt()
        # (B, yxhwsc, H, W)
        out_boxes = torch.stack([b_y / ih, b_x / iw, b_h / ih, b_w / iw, b_dir[:, 0], b_dir[:, 1]], 1)
        return out_boxes


class VerticalLayerSimple(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        c_in = c_out = num_filters * 2
        self.net = nn.Sequential(
            nn.InstanceNorm2d(c_in, affine=True),
            nn.ReLU(True),
            ShuffleConv2d(c_in, c_in, 3, 1, 1, bias=False, groups=4),

            nn.InstanceNorm2d(c_in, affine=True),
            nn.ReLU(True),
            ShuffleConv2d(c_in, c_out, 3, 1, 1, bias=False, groups=4),
        )

    def forward(self, input_layers):
        output_layers = []
        cx = Variable(input_layers[0].data.new(input_layers[0].shape).zero_())
        for input in input_layers:
            if cx.shape[-1] // 2 == input.shape[-1]:
                cx = F.max_pool2d(cx, 3, 2, 1)
            elif cx.shape[-1] * 2 == input.shape[-1]:
                cx = F.upsample(cx, scale_factor=2)
            elif cx.shape[-1] != input.shape[-1]:
                raise ValueError((cx.shape, input.shape))
            input = torch.cat([input, cx], 1)
            cx_new, hx = self.net(input).chunk(2, 1)
            cx = cx + cx_new
            output_layers.append(hx)
        return output_layers


class TopDownLayer(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.top_down = VerticalLayerSimple(num_filters)

    def forward(self, x):
        return self.top_down(x[::-1])[::-1]


class BottomUpLayer(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.bottom_up = VerticalLayerSimple(num_filters)

    def forward(self, x):
        return self.bottom_up(x)


class BidirectionalLayer(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.layers = nn.ModuleList([
            TopDownLayer(num_filters),
            BottomUpLayer(num_filters),
            TopDownLayer(num_filters),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = [a + b for a, b in zip(x, layer(x))]
        return x


class AdaptiveFeaturePooling(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        x = x.view(-1, self.num_groups, *x.shape[1:])
        x = x.max(1)[0]
        return x


class BatchChannels(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters

    def forward(self, x):
        return x.contiguous().view(-1, self.num_filters, *x.shape[2:])


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out=None):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        self.shortcut = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else (lambda x: x)
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(c_in, affine=True),
            nn.ReLU(True),
            ShuffleConv2d(c_in, c_in, 3, 1, 1, bias=False, groups=4),

            nn.InstanceNorm2d(c_in, affine=True),
            nn.ReLU(True),
            ShuffleConv2d(c_in, c_out, 3, 1, 1, bias=False, groups=4),
        )

    def forward(self, input):
        x = self.layers(input)
        x += self.shortcut(input)
        return x


def get_features(network):
    if network == 'resnet50':
        rn = resnet50(True)
        features = nn.ModuleList([
            nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1),
            rn.layer2,
            rn.layer3,
            rn.layer4
        ])
        channels = 256, 512, 1024, 2048
    elif network == 'resnext101_32x4d':
        features = nn.ModuleList(list(resnext101_32x4d().features)[:8])
        features = nn.ModuleList([
            nn.Sequential(*features[:5]),
            features[5],
            features[6],
            features[7],
        ])
        channels = 256, 512, 1024, 2048
    elif network == 'dpn68b':
        dpn = list(dpn68b().features)
        features = nn.ModuleList([
            nn.Sequential(*dpn[:4]),
            nn.Sequential(*dpn[4:8]),
            nn.Sequential(*dpn[8:20]),
            nn.Sequential(*dpn[20:23])
        ])
        channels = 144, 320, 704, 832
    elif network == 'dpn92':
        dpn = list(dpn92().features)
        features = nn.ModuleList([
            nn.Sequential(*dpn[:4]),
            nn.Sequential(*dpn[4:8]),
            nn.Sequential(*dpn[8:28]),
            nn.Sequential(*dpn[28:31])
        ])
        channels = 336, 704, 1552, 2688
    else:
        raise ValueError(network)
    return features, channels


class FPN(nn.Module):
    mask_size = 28
    region_size = 7
    num_feature_groups = 4

    def __init__(self, out_image_channels=0, num_filters=256, enable_bidir=True):
        super().__init__()
        self.enable_bidir = enable_bidir

        self.mask_pixel_sizes = (1, 2, 4)
        self.mask_strides = (4, 8, 16)
        self.resnet, lat_channels = get_features('dpn92')

        # Top layer
        self.toplayer = nn.Conv2d(lat_channels[3], num_filters, kernel_size=1, stride=1, padding=0, bias=False)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(lat_channels[2], num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.latlayer2 = nn.Conv2d(lat_channels[1], num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.latlayer3 = nn.Conv2d(lat_channels[0], num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        # self.latlayer4 = nn.Conv2d(64, num_filters, kernel_size=1, stride=1, padding=0, bias=False)

        if self.enable_bidir:
            self.bidir = BidirectionalLayer(num_filters)
        else:
            self.bidir = None

        if out_image_channels != 0:
            self.img_net = nn.Sequential(
                BatchChannels(num_filters),
                ResBlock(num_filters),
                AdaptiveFeaturePooling(FPN.num_feature_groups),

                ShuffleConv2d(num_filters, num_filters // 2, 3, 1, 1, bias=False, groups=4),
                nn.InstanceNorm2d(num_filters // 2, affine=True),
                nn.ReLU(True),

                nn.Upsample(scale_factor=2),

                ShuffleConv2d(num_filters // 2, num_filters // 4, 3, 1, 1, bias=False, groups=4),
                nn.InstanceNorm2d(num_filters // 4, affine=True),
                nn.ReLU(True),

                nn.Upsample(scale_factor=2),

                ShuffleConv2d(num_filters // 4, num_filters // 8, 3, 1, 1, bias=False, groups=4),
                nn.InstanceNorm2d(num_filters // 8, affine=True),
                nn.ReLU(True),

                nn.Conv2d(num_filters // 8, out_image_channels, 1),
            )
            init_fg_conf = 0.01
            self.img_net[-1].bias.data.fill_(-math.log((1 - init_fg_conf) / init_fg_conf))
        else:
            self.img_net = None

        self.mask_head = MaskHead(num_filters, self.region_size, self.mask_size)
        self.box_head = BoxHead(num_filters, self.region_size)
        self.score_head = ScoreHead(num_filters, 1)

    def freeze_pretrained_layers(self, freeze):
        self.resnet.train(not freeze)
        for module in self.resnet.modules():
            if isinstance(module, _InstanceNorm):
                module.use_running_stats = freeze
        for p in self.resnet.parameters():
            p.requires_grad = not freeze

    def combine_levels(self, levels, output_indexes):
        combined_levels = []
        for idx, base_level in enumerate(levels):
            if idx not in output_indexes:
                continue

            resized_levels = []
            bsh = base_level.shape[2]
            for other_level in levels:
                osh = other_level.shape[2]
                assert other_level.dim() == 4
                assert other_level.shape[2] / other_level.shape[3] == base_level.shape[2] / base_level.shape[3]
                if osh > bsh:
                    other_level = F.avg_pool2d(other_level, osh // bsh)
                elif osh < bsh:
                    other_level = F.upsample(other_level, scale_factor=bsh // osh, mode='bilinear')
                resized_levels.append(other_level)

            resized_levels = torch.cat(resized_levels, 1)

            combined_levels.append(resized_levels)
        return combined_levels

    def forward(self, x, output_unpadding=0):
        make_features_volatile = not self.resnet.training and not x.volatile
        if make_features_volatile:
            x = Variable(x.data, volatile=True)

        c2 = self.resnet[0](x)
        c3 = self.resnet[1](c2)
        c4 = self.resnet[2](c3)
        c5 = self.resnet[3](c4)

        c2, c3, c4, c5 = [torch.cat(t, 1) for t in (c2, c3, c4, c5)]

        if make_features_volatile:
            c2, c3, c4, c5 = [Variable(v.data) for v in (c2, c3, c4, c5)]

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self.latlayer1(c4)
        p3 = self.latlayer2(c3)
        p2 = self.latlayer3(c2)
        # p1 = self.latlayer4(c1)

        del x, c5, c4, c3, c2

        if self.enable_bidir:
            p2, p3, p4, p5 = self.bidir((p2, p3, p4, p5))

        p2, p3, p4 = self.combine_levels((p2, p3, p4, p5), (0, 1, 2))

        img = unpad(self.img_net(p2), output_unpadding) if self.img_net is not None else None

        def mask_score_box(fmap, unpadding):
            return self.mask_head(fmap, unpadding), self.score_head(fmap, unpadding), self.box_head(fmap, unpadding)

        m4 = mask_score_box(p4, output_unpadding // 16)
        m3 = mask_score_box(p3, output_unpadding // 8)
        m2 = mask_score_box(p2, output_unpadding // 4)

        return (m2, m3, m4), img

    def predict_masks(self, x):
        return self.mask_head.predict_masks(x)
