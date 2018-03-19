# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.utils import model_zoo
from torchvision.models.resnet import resnet50, model_urls
from ..skip_connections import ResidualSequential, DenseSequential
import itertools
import math
from pretrainedmodels import resnext101_32x4d, resnext101_64x4d


# def weights_init(m):
#     # if isinstance(m, F._ConvNd) or isinstance(m, nn.Linear):
#     #     torch.nn.init.orthogonal(m.weight.data, torch.nn.init.calculate_gain('relu'))
#     #     if m.bias is not None:
#     #         m.bias.data.fill_(0)
#     if isinstance(m, _BatchNorm) or isinstance(m, _InstanceNorm):
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


class MaskHead(nn.Module):
    def __init__(self, num_filters, region_size, mask_size):
        super().__init__()
        assert mask_size % region_size == 0
        self.num_filters = num_filters
        self.region_size = region_size
        self.mask_size = mask_size

        # self.preproc_layers = nn.Sequential(
        #     # nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
        #     # nn.BatchNorm2d(num_filters, affine=True),
        #     # nn.ReLU(True),
        #     # nn.Sequential(
        #     #     nn.BatchNorm2d(num_filters, affine=True),
        #     #     nn.ReLU(True),
        #     #     nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
        #     #     nn.BatchNorm2d(num_filters, affine=True),
        #     #     nn.ReLU(True),
        #     #     nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
        #     # ),
        # )
        self.mask_layers = [
            ResidualSequential(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
            )
        ]
        cur_size = region_size
        cur_filters = num_filters
        while cur_size != mask_size:
            prev_filters = cur_filters
            cur_filters //= 2
            cur_size *= 2
            self.mask_layers.append(nn.Sequential(
                nn.Conv2d(prev_filters, cur_filters, 1, bias=False),
                nn.Upsample(scale_factor=2),
                ResidualSequential(
                    nn.Sequential(
                        nn.Conv2d(cur_filters, cur_filters, 1, bias=False),
                        nn.BatchNorm2d(cur_filters, affine=True),
                        nn.ReLU(True),
                        nn.Conv2d(cur_filters, cur_filters, 3, 1, 1, bias=False),
                    ),
                    nn.Sequential(
                        nn.Conv2d(cur_filters, cur_filters, 1, bias=False),
                        nn.BatchNorm2d(cur_filters, affine=True),
                        nn.ReLU(True),
                        nn.Conv2d(cur_filters, cur_filters, 3, 1, 1, bias=False),
                    ),
                )
            ))
        self.mask_layers.append(nn.Sequential(
            nn.Conv2d(cur_filters, cur_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cur_filters, affine=True),
            nn.ReLU(True),
            nn.Conv2d(cur_filters, 1, 1),
        ))
        self.mask_layers = nn.Sequential(*self.mask_layers)

    def forward(self, input):
        return input.contiguous() # self.preproc_layers(input.contiguous())

    def predict_masks(self, x):
        assert x.shape == (x.shape[0], self.num_filters, self.region_size, self.region_size)
        return self.mask_layers(x.contiguous())


class ScoreHead(nn.Module):
    def __init__(self, num_filters, num_scores, init_foreground_confidence=0.01):
        super().__init__()
        self.score_layers = nn.Sequential(
            ResidualSequential(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
            ),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, num_scores, 1),
        )
        self.score_layers[-1].bias.data.fill_(-math.log((1 - init_foreground_confidence) / init_foreground_confidence))

    def forward(self, input):
        score = self.score_layers(input.contiguous())
        return score


class BoxHead(nn.Module):
    def __init__(self, num_filters, region_size,
                 sizes=((2 ** -0.5, 2 * 2 ** -0.5), (1, 1), (2 * 2 ** -0.5, 2 ** -0.5)),
                 scales=(2**(-1 / 3), 1, 2**(1 / 3))):
        super().__init__()
        self.register_buffer('pixel_boxes', None)
        pixel_boxes = [(size[0] * scale * region_size, size[1] * scale * region_size)
                            for size, scale in itertools.product(sizes, scales)]
        self.pixel_boxes = torch.Tensor(pixel_boxes).float()
        self.layers = nn.Sequential(
            ResidualSequential(
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, 1, bias=False),
                    nn.BatchNorm2d(num_filters, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
                ),
            ),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, len(self.pixel_boxes) * 4, 1, bias=False),
        )
        self.layers[-1].weight.data.mul_(0.2)

    def forward(self, input):
        ih, iw = input.shape[2:]
        boxes = self.layers(input.contiguous())
        # boxes = Variable(boxes.data.fill_(0), requires_grad=True)
        boxes = boxes.view(boxes.shape[0], len(self.pixel_boxes), 4, *boxes.shape[2:])
        anchor_sizes = Variable(self.pixel_boxes.view(1, len(self.pixel_boxes), 2, 1, 1))
        anchor_pos_y = torch.arange(ih).type_as(input.data).view(1, 1, -1, 1).add_(0.5)
        anchor_pos_x = torch.arange(iw).type_as(input.data).view(1, 1, 1, -1).add_(0.5)
        anchor_pos_y, anchor_pos_x = Variable(anchor_pos_y), Variable(anchor_pos_x)
        sqrt2 = Variable(boxes.data.new([2]))
        b_h = sqrt2.pow(boxes[:, :, 2]) * anchor_sizes[:, :, 0]
        b_w = sqrt2.pow(boxes[:, :, 3]) * anchor_sizes[:, :, 1]
        b_y = boxes[:, :, 0] * anchor_sizes[:, :, 0] + anchor_pos_y - b_h / 2
        b_x = boxes[:, :, 1] * anchor_sizes[:, :, 1] + anchor_pos_x - b_w / 2
        # (B, yxhw, NBox, H, W)
        out_boxes = torch.stack([b_y / ih, b_x / iw, b_h / ih, b_w / iw], 1)
        anchor_part_shape = 1, len(self.pixel_boxes), anchor_pos_y.numel(), anchor_pos_x.numel()
        anchor_h = anchor_sizes.data[:, :, 0]
        anchor_w = anchor_sizes.data[:, :, 1]
        # (yxhw, NBox, H, W)
        anchor_boxes = torch.cat([
            anchor_pos_y.data.sub(anchor_h / 2).expand(*anchor_part_shape),
            anchor_pos_x.data.sub(anchor_w / 2).expand(*anchor_part_shape),
            anchor_h.expand(*anchor_part_shape),
            anchor_w.expand(*anchor_part_shape),
        ], 0)
        anchor_boxes.div_(self.pixel_boxes.new([ih, iw, ih, iw]).view(4, 1, 1, 1))
        return out_boxes, anchor_boxes


class VerticalLayerSimple(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_filters * 2, affine=True),
            nn.ReLU(True),
            nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),
            nn.Conv2d(num_filters, num_filters * 2, 3, 1, 1, bias=False),
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
            cx += cx_new
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


class GroupMaxout(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        x = x.view(x.shape[0], self.groups, -1, *x.shape[2:])
        x = x.max(1)[0]
        return x


class FPN(nn.Module):
    mask_size = 28
    region_size = 7

    def __init__(self, out_image_channels=0, num_filters=256, enable_bidir=False):
        super().__init__()
        self.enable_bidir = enable_bidir

        self.mask_pixel_sizes = (1, 2, 4)
        self.mask_strides = (4, 8, 16)
        self.resnet = resnext101_32x4d()

        # Top layer
        self.toplayer = nn.Conv2d(2048, num_filters, kernel_size=1, stride=1, padding=0, bias=False)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.latlayer2 = nn.Conv2d(512, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        self.latlayer3 = nn.Conv2d(256, num_filters, kernel_size=1, stride=1, padding=0, bias=False)
        # self.latlayer4 = nn.Conv2d(64, num_filters, kernel_size=1, stride=1, padding=0, bias=False)

        if self.enable_bidir:
            self.bidir = BidirectionalLayer(num_filters)
        else:
            self.bidir = None

        num_comb_layers = 4
        ncf = num_filters * num_comb_layers
        self.combiner = nn.Sequential(
            ResidualSequential(
                nn.Sequential(
                    nn.Conv2d(ncf, ncf, 1, groups=num_comb_layers, bias=False),
                    nn.BatchNorm2d(num_filters * num_comb_layers, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(ncf, ncf, 3, 1, 1, groups=num_comb_layers, bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(ncf, ncf, 1, groups=num_comb_layers, bias=False),
                    nn.BatchNorm2d(num_filters * num_comb_layers, affine=True),
                    nn.ReLU(True),
                    nn.Conv2d(ncf, ncf, 3, 1, 1, groups=num_comb_layers, bias=False),
                ),
            ),
            GroupMaxout(num_comb_layers),
        )

        self.img_net = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters // 2, affine=True),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(num_filters // 2, num_filters // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters // 4, affine=True),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(num_filters // 4, num_filters // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters // 8, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters // 8, out_image_channels, 1),
        ) if out_image_channels != 0 else None

        self.mask_head = MaskHead(num_filters, self.region_size, self.mask_size)
        self.box_head = BoxHead(num_filters, self.region_size)
        self.score_head = ScoreHead(num_filters, len(self.box_head.pixel_boxes))

    def freeze_pretrained_layers(self, freeze):
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
                assert other_level.shape[2] == other_level.shape[3]
                if osh > bsh:
                    other_level = F.max_pool2d(other_level, osh // bsh)
                elif osh < bsh:
                    other_level = F.upsample(other_level, scale_factor=bsh // osh, mode='bilinear')
                resized_levels.append(other_level)
            resized_levels = torch.cat(resized_levels, 1)
            assert resized_levels.shape[1] == base_level.shape[1] * len(levels)
            combined_levels.append(resized_levels)
        return combined_levels

    def forward(self, x, output_unpadding=0):
        # c1 = self.resnet.conv1(x)
        # c1 = self.resnet.bn1(c1)
        # c1 = self.resnet.relu(c1)
        # c1 = self.resnet.maxpool(c1)
        #
        # c2 = self.resnet.layer1(c1)
        # c3 = self.resnet.layer2(c2)
        # c4 = self.resnet.layer3(c3)
        # c5 = self.resnet.layer4(c4)

        c1 = self.resnet.features[0](x)
        c1 = self.resnet.features[1](c1)
        c1 = self.resnet.features[2](c1)
        c1 = self.resnet.features[3](c1)

        c2 = self.resnet.features[4](c1)
        c3 = self.resnet.features[5](c2)
        c4 = self.resnet.features[6](c3)
        c5 = self.resnet.features[7](c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self.latlayer1(c4)
        p3 = self.latlayer2(c3)
        p2 = self.latlayer3(c2)
        # p1 = self.latlayer4(c1)

        if self.enable_bidir:
            p2, p3, p4, p5 = self.bidir((p2, p3, p4, p5))

        p2, p3, p4 = self.combine_levels((p2, p3, p4, p5), (0, 1, 2))
        p2, p3, p4 = [self.combiner(x) for x in (p2, p3, p4)]

        img = self.img_net(p2)[:, :, output_unpadding:-output_unpadding, output_unpadding:-output_unpadding] \
            if self.img_net is not None else None

        if output_unpadding != 0:
            assert output_unpadding % 32 == 0
            ou = output_unpadding
            p5, p4, p3, p2 = [p[:, :, div:-div, div:-div] for (p, div) in
                              ((p5, ou // 32), (p4, ou // 16), (p3, ou // 8), (p2, ou // 4))]

        m4 = self.mask_head(p4), self.score_head(p4), self.box_head(p4)
        m3 = self.mask_head(p3), self.score_head(p3), self.box_head(p3)
        m2 = self.mask_head(p2), self.score_head(p2), self.box_head(p2)

        return (m2, m3, m4), img

    def predict_masks(self, x):
        return self.mask_head.predict_masks(x)
