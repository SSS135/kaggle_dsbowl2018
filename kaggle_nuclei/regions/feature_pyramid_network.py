# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.utils import model_zoo
from torchvision.models.resnet import resnet50, model_urls


# def weights_init(m):
#     # if isinstance(m, F._ConvNd) or isinstance(m, nn.Linear):
#     #     torch.nn.init.orthogonal(m.weight.data, torch.nn.init.calculate_gain('relu'))
#     #     if m.bias is not None:
#     #         m.bias.data.fill_(0)
#     if isinstance(m, _BatchNorm) or isinstance(m, _InstanceNorm):
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


class MaskHead(nn.Module):
    conv_size = 4

    def __init__(self, num_filters):
        super().__init__()
        assert FPN.mask_size % self.conv_size == 0
        self.num_filters = num_filters

        self.preproc_layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),
            # nn.Sequential(
            #     nn.BatchNorm2d(num_filters, affine=True),
            #     nn.ReLU(True),
            #     nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #     nn.BatchNorm2d(num_filters, affine=True),
            #     nn.ReLU(True),
            #     nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            # ),
        )
        self.mask_layers = []
        cur_size = self.conv_size
        cur_filters = num_filters
        while cur_size != FPN.mask_size:
            prev_filters = cur_filters
            cur_filters //= 2
            cur_size *= 2
            self.mask_layers.append(nn.Sequential(
                # nn.Conv2d(prev_filters, cur_filters, 1, bias=False),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(prev_filters, cur_filters, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cur_filters, affine=True),
                nn.ReLU(True),
                # ResidualSequential(
                #     nn.Sequential(
                #         nn.BatchNorm2d(cur_filters, affine=True),
                #         nn.ReLU(True),
                #         nn.Conv2d(cur_filters, cur_filters, 3, 1, 1, bias=False),
                #         nn.BatchNorm2d(cur_filters, affine=True),
                #         nn.ReLU(True),
                #         nn.Conv2d(cur_filters, cur_filters, 3, 1, 1, bias=False),
                #     ),
                # ),
            ))
        self.mask_layers.append(nn.Sequential(
            nn.Conv2d(cur_filters, 1, 1),
        ))
        self.mask_layers = nn.Sequential(*self.mask_layers)

    def forward(self, input):
        return self.preproc_layers(input.contiguous())

    def predict_masks(self, x):
        assert x.shape == (x.shape[0], self.num_filters, self.conv_size, self.conv_size)
        return self.mask_layers(x.contiguous())


class ScoreHead(nn.Module):
    def __init__(self, num_filters, num_scores):
        super().__init__()
        assert FPN.mask_size % 4 == 0

        self.score_layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),

            nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.ReLU(True),
            # ResidualSequential(
            #     nn.Sequential(
            #         nn.BatchNorm2d(num_filters, affine=True),
            #         nn.ReLU(True),
            #         nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #         nn.BatchNorm2d(num_filters, affine=True),
            #         nn.ReLU(True),
            #         nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #     ),
            #     nn.Sequential(
            #         nn.BatchNorm2d(num_filters, affine=True),
            #         nn.ReLU(True),
            #         nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #         nn.BatchNorm2d(num_filters, affine=True),
            #         nn.ReLU(True),
            #         nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #     ),
            #     nn.Sequential(
            #         nn.BatchNorm2d(num_filters, affine=True),
            #         nn.ReLU(True),
            #         nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #         nn.BatchNorm2d(num_filters, affine=True),
            #         nn.ReLU(True),
            #         nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False),
            #     ),
            # ),
            nn.Conv2d(num_filters, num_scores, 4),
        )

    def forward(self, input):
        score = self.score_layers(input.contiguous())
        return score


class VerticalLayerSimple(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm2d(num_filters * 2),
            nn.BatchNorm2d(num_filters * 2, affine=True),
            nn.ReLU(True),
            nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(num_filters),
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


class FPN(nn.Module):
    mask_size = 32
    mask_kernel_size = 4

    def __init__(self, num_scores=1, out_image_channels=0, num_filters=256):
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
        self.latlayer4 = nn.Conv2d(64, num_filters, kernel_size=1, stride=1, padding=0)

        self.bidir = BidirectionalLayer(num_filters)

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

        self.mask_head = MaskHead(num_filters)
        self.score_head = ScoreHead(num_filters, num_scores)
        self.conv_size = self.mask_head.conv_size

        # self.reset_weights()

    # def reset_weights(self):
    #     self.apply(weights_init)
    #     self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    def freeze_pretrained_layers(self, freeze):
        for p in self.resnet.parameters():
            p.requires_grad = not freeze

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
        p4 = self.latlayer1(c4)
        p3 = self.latlayer2(c3)
        p2 = self.latlayer3(c2)
        p1 = self.latlayer4(c1)

        p1, p2, p3, p4, p5 = self.bidir((p1, p2, p3, p4, p5))

        img = self.img_net(p1)[:, :, output_unpadding:-output_unpadding, output_unpadding:-output_unpadding] \
            if self.img_net is not None else None

        if output_unpadding != 0:
            assert output_unpadding % 32 == 0
            ou = output_unpadding
            p5, p4, p3, p2 = [p[:, :, div:-div, div:-div] for (p, div) in
                              ((p5, ou // 32), (p4, ou // 16), (p3, ou // 8), (p2, ou // 4))]

        m5 = self.mask_head(p5), self.score_head(p5)
        m4 = self.mask_head(p4), self.score_head(p4)
        m3 = self.mask_head(p3), self.score_head(p3)
        m2 = self.mask_head(p2), self.score_head(p2)

        return (m2, m3, m4, m5), img

    def predict_masks(self, x):
        return self.mask_head.predict_masks(x)
