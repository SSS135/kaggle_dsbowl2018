import copy
import math
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from optfn.cosine_annealing import CosineAnnealingRestartLR
from optfn.gadam import GAdam
from optfn.param_groups_getter import get_param_groups
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn

from .dataset import NucleiDataset
from .feature_pyramid_network import FPN
from .iou import threshold_iou, iou
from .losses import dice_loss, soft_dice_loss, clipped_mse_loss
from .unet import UNet
from .ms_d_net import MSDNet
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm


def train_preprocessor_gan(train_data, epochs=15, pretrain_epochs=7, resnet=False):
    dataset = NucleiDataset(train_data, supersample=1)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4, pin_memory=True)

    pad = dataloader.dataset.padding
    # mask_channels = 2

    # if resnet:
    #     gen_model = FPN(3).cuda()
    #     gen_optimizer = GAdam(get_param_groups(gen_model), lr=1e-4, betas=(0.9, 0.999), nesterov=0.0,
    #                           weight_decay=5e-4, avg_sq_mode='tensor', amsgrad=False)
    # else:
    gen_model = UNet(3, 3).cuda()
    gen_optimizer = GAdam(get_param_groups(gen_model), lr=1e-4, betas=(0.5, 0.999),
                          amsgrad=False, nesterov=0.5, weight_decay=1e-5, norm_weight_decay=False)

    disc_model = GanD(6).cuda()
    disc_optimizer = GAdam(get_param_groups(disc_model), lr=1e-4, betas=(0.5, 0.999),
                           amsgrad=False, nesterov=0.5, weight_decay=1e-5, norm_weight_decay=False)

    gen_model.apply(weights_init)
    disc_model.apply(weights_init)

    gen_scheduler = CosineAnnealingRestartLR(gen_optimizer, len(dataloader), 2)
    disc_scheduler = CosineAnnealingRestartLR(disc_optimizer, len(dataloader), 2)

    best_model = gen_model
    best_score = -math.inf

    sys.stdout.flush()

    one = Variable(torch.cuda.FloatTensor([0.95]))
    zero = Variable(torch.cuda.FloatTensor([0.05]))

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            if resnet:
                gen_model.freeze_pretrained_layers(epoch < pretrain_epochs)
            t_iou_ma, f_iou_ma, sdf_loss_ma, cont_loss_ma = 0, 0, 0, 0
            for i, (img, mask, sdf, obj_info) in enumerate(pbar):
                x_train = Variable(img.cuda())
                sdf_train = Variable(sdf.cuda())
                mask_train = mask.cuda()
                x_train_unpad = x_train[:, :, pad:-pad, pad:-pad]

                # mask_train = remove_missized_objects(mask_train, 0.1, 0.6)
                mask_target = (mask_train > 0).float().clamp(min=0.05, max=0.95)
                mask_target = Variable(mask_target)

                # mask_target = (mask_train.data > 0).float() * 2 - 1
                # mask_target = mask_target.mul_(3).add_(0.5 * mask_target.clone().normal_(0, 1))
                # mask_target = F.sigmoid(Variable(mask_target))

                # split_mask = split_labels(mask_train.data, mask_channels).float()
                # split_mask = split_mask.sub_(0.5).mul_(6).add_(0.5 * split_mask.clone().normal_(0, 1))
                # split_mask = F.softmax(Variable(split_mask), 1)
                # split_mask = split_mask.clamp(min=0.05, max=0.95) + split_mask.clone().uniform_(-0.05, 0.05)
                # split_mask /= split_mask.sum(1, keepdim=True)
                # split_mask = Variable(split_mask)

                cont_train = 1 - sdf_train.data ** 2
                cont_train = (cont_train.clamp(0.9, 1) - 0.9) * 20 - 1
                cont_train = Variable(cont_train)

                # discriminator
                disc_optimizer.zero_grad()

                # for p in disc_model.parameters():
                #     p.data.clamp_(-0.01, 0.01)

                real_input = torch.cat([mask_target, sdf_train, cont_train, x_train_unpad], 1)
                real_d, real_features = disc_model(real_input)
                loss_real = 0
                # loss_real += -real_d.mean()
                loss_real += F.binary_cross_entropy_with_logits(real_d, one.expand_as(real_d))
                # loss_real += 0.5 * (1 - real_d.clamp(max=1)).pow_(2).mean()
                loss_real.backward()

                # gen_noise = Variable(x_train.data.new(x_train.shape[0], 1, *x_train.shape[2:]).normal_(0, 1))
                # gen_input = torch.cat([x_train, gen_noise], 1)
                out = gen_model(x_train)[:, :, pad:-pad, pad:-pad]
                out_sdf, out_cont, out_mask = out[:, 0:1], out[:, 1:2], out[:, 2:3]
                out_mask, out_sdf = F.sigmoid(out_mask), out_sdf.contiguous()

                fake_input = torch.cat([out_mask, out_sdf, out_cont, x_train_unpad], 1)
                fake_d, fake_features = disc_model(fake_input.detach())
                loss_fake = 0
                # loss_fake += fake_d.mean()
                loss_fake += F.binary_cross_entropy_with_logits(fake_d, zero.expand_as(fake_d))
                # loss_fake += 0.5 * (-1 - fake_d.clamp(min=-1)).pow_(2).mean()
                loss_fake.backward()

                disc_optimizer.step()

                # generator
                gen_optimizer.zero_grad()

                gen_d, fake_features = disc_model(fake_input)
                loss_gen = 0
                # loss_gen += -gen_d.mean()
                # loss_gen += F.binary_cross_entropy_with_logits(gen_d, one.expand_as(gen_d))
                # loss_gen += 0.5 * (1 - gen_d.div(3).clamp(min=-1)).pow_(2).mean()
                loss_gen += F.mse_loss(fake_features, real_features.detach())
                # loss_gen += F.mse_loss(gen_d, real_d.detach())

                sdf_loss = F.mse_loss(out_sdf, sdf_train)
                cont_loss = F.mse_loss(out_cont, cont_train)
                # mask_loss = soft_dice_loss(out_mask, mask_target)
                loss = loss_gen + sdf_loss + cont_loss # sdf_loss.mean() + cont_loss.mean() #+ mask_loss
                loss.backward()

                gen_optimizer.step()

                gen_scheduler.step()
                disc_scheduler.step()

                # flat_out_mask = (out_mask.max(1)[1] != out_mask.shape[1] - 1).float()
                f_iou = iou(out_mask.data, mask_train)
                t_iou = threshold_iou(f_iou)
                bc = 1 - 0.99 ** (i + 1)
                sdf_loss_ma = 0.99 * sdf_loss_ma + 0.01 * sdf_loss.data[0]
                cont_loss_ma = 0.99 * cont_loss_ma + 0.01 * cont_loss.data[0]
                t_iou_ma = 0.99 * t_iou_ma + 0.01 * t_iou.mean()
                f_iou_ma = 0.99 * f_iou_ma + 0.01 * f_iou.mean()
                pbar.set_postfix(epoch=epoch, T_IoU=t_iou_ma / bc, SDF=sdf_loss_ma / bc,
                                 Cont=cont_loss_ma / bc, IOU=f_iou_ma / bc, refresh=False)

            if t_iou_ma > best_score:
                best_score = t_iou_ma
                best_model = copy.deepcopy(gen_model)
    return best_model


def remove_missized_objects(mask, min_size, max_size):
    mask = mask.clone()
    obj_count = mask.max()
    mask_size = math.sqrt(mask.shape[2] * mask.shape[3])
    for idx in range(1, obj_count + 1):
        obj_mask = mask == idx
        size = math.sqrt(obj_mask.sum()) / mask_size
        if size < min_size or size > max_size:
            mask[obj_mask] = 0
    return mask


# custom weights initialization called on netG and netD
def weights_init(m):
    # if isinstance(m, _ConvNd) or isinstance(m, nn.Linear):
    #     # m.weight.data.normal_(0.0, 0.02)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0)
    if isinstance(m, _BatchNorm):
        m.weight.data.normal_(1.0, 0.05)
        m.bias.data.fill_(0)


def split_labels(mask, num_split_channels):
    assert num_split_channels >= 2
    mask_label_count = mask.max()
    split_mask = mask.new(mask.shape[0], num_split_channels, *mask.shape[2:]).byte().fill_(0)
    if mask_label_count == 0:
        return split_mask
    c_idx = mask.new(mask_label_count).random_(num_split_channels - 1)
    split_mask[:, -1] = (mask == 0).squeeze(1)
    for mc in range(mask_label_count):
        split_mask[:, c_idx[mc]] |= (mask == mc + 1).squeeze(1)
    return split_mask


class GanD(nn.Module):
    def __init__(self, nc=3, nf=64):
        super().__init__()
        self.head = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
        )
        self.tail = nn.Sequential(
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        features = self.head(input)
        output = self.tail(features)
        return output.view(output.shape[0], -1).mean(-1), features


class GanD_UNet(nn.Module):
    def __init__(self, nc=3, nf=64):
        super().__init__()
        self.unet = UNet(nc, 1, f=nf)
        # self.conv = nn.Conv2d(1, 1, 8, 4)
        # self.head = nn.Sequential(
        #     nn.Linear(nf * 2, nf * 2),
        #     nn.BatchNorm2d(nf * 2),
        #     nn.ReLU(True),
        #     nn.Linear(nf * 2, 1),
        # )

    def forward(self, input):
        features = self.unet(input)
        # output = features
        # while output.shape[2] >= self.conv.kernel_size[0]:
        #     output = self.conv(output)
        output = features.view(input.shape[0], -1).mean()
        # output = features.view(*features.shape[:2], -1)
        # output = torch.cat([output.mean(-1), output.std(-1)], 1)
        # output = self.head(output).view(-1)
        return output, features