import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from optfn.cosine_annealing import CosineAnnealingRestartParam
from optfn.gadam import GAdam
from optfn.param_groups_getter import get_param_groups
from skimage.morphology import label
from torch.autograd import Variable
from tqdm import tqdm

from .dataset import train_pad
from .iou import mean_threshold_object_iou, iou
from .postprocessing_dataset import PostprocessingDataset
from .unet import UNet

try:
    import ppo_pytorch.ppo_pytorch as rl
except:
    import ppo_pytorch as rl


def train_postprocessor_ppo(train_data, train_pred, epochs=7):
    dataset = PostprocessingDataset(train_data, train_pred)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=4, pin_memory=True, drop_last=True)

    gen_model = UNet(6, 2).cuda()
    gen_optimizer = GAdam(get_param_groups(gen_model), lr=1e-4, betas=(0.5, 0.999), avg_sq_mode='tensor', weight_decay=5e-5)
    disc_model = GanD_UNet(8).cuda()
    disc_optimizer = GAdam(get_param_groups(disc_model), lr=1e-4, betas=(0.5, 0.999), avg_sq_mode='tensor', weight_decay=5e-5)

    gen_scheduler = CosineAnnealingRestartParam(gen_optimizer, len(dataloader), 2)
    disc_scheduler = CosineAnnealingRestartParam(disc_optimizer, len(dataloader), 2)

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            src_iou_ma, aug_iou_ma, fake_loss_ma, src_obj_iou_ma, aug_obj_iou_ma = 0, 0, 0, 0, 0
            for i, batch in enumerate(pbar):
                img, pred_mask, pred_sdf, pred_cont, mask = [x.cuda() for x in batch]
                pred_mask = torch.sigmoid(pred_mask)
                pred_unpad_mask = pred_mask[:, :, train_pad:-train_pad, train_pad:-train_pad]
                img, pred_mask, pred_sdf, pred_cont = [Variable(x) for x in (img, pred_mask, pred_sdf, pred_cont)]

                # disc
                disc_optimizer.zero_grad()

                gen_in = torch.cat([img, pred_mask - 0.5, pred_sdf, pred_cont], 1)
                gen_in_unpad = gen_in[:, :, train_pad:-train_pad, train_pad:-train_pad]

                src_iou = iou(pred_unpad_mask, mask)
                src_obj_iou = object_iou(pred_unpad_mask, mask)

                gen_out = gen_model(gen_in)
                gen_out = F.softmax(gen_out, 1)
                gen_out_unpad = gen_out[:, :, train_pad:-train_pad, train_pad:-train_pad]
                # gen_mask_unpad = gen_out_unpad[:, 1:].data # gen_out_unpad.data + pred_unpad_mask

                gm_max = gen_out_unpad.data.max(1, keepdim=True)[1]
                aug_iou = iou((gm_max != 0).float(), mask)
                aug_obj_iou = multilayer_object_iou(gen_out_unpad.data, mask)

                disc_in = torch.cat([gen_in_unpad, gen_out_unpad - 1 / gen_out_unpad.shape[1]], 1)
                disc_out, _ = disc_model(disc_in.detach())
                fake_disc_loss = F.mse_loss(disc_out, Variable(aug_iou + aug_obj_iou))
                fake_disc_loss.backward()
                disc_optimizer.step()

                # gen
                gen_optimizer.zero_grad()

                disc_out, _ = disc_model(disc_in)
                gen_loss = -disc_out.mean()
                gen_loss.backward()
                gen_optimizer.step()

                gen_scheduler.step()
                disc_scheduler.step()

                bc = 1 - 0.99 ** (i + 1)
                src_iou_ma = 0.99 * src_iou_ma + 0.01 * src_iou.mean()
                aug_iou_ma = 0.99 * aug_iou_ma + 0.01 * aug_iou.mean()
                src_obj_iou_ma = 0.99 * src_obj_iou_ma + 0.01 * src_obj_iou.mean()
                aug_obj_iou_ma = 0.99 * aug_obj_iou_ma + 0.01 * aug_obj_iou.mean()
                fake_loss_ma = 0.99 * fake_loss_ma + 0.01 * fake_disc_loss.data[0] ** 0.5
                pbar.set_postfix(EP=epoch,
                                 SIOU=src_iou_ma / bc, AIOU=aug_iou_ma / bc,
                                 SOIOU=src_obj_iou_ma / bc, AOIOU=aug_obj_iou_ma / bc,
                                 L=fake_loss_ma / bc, refresh=False)

    return gen_model


# def train_postprocessor_ppo_const(train_data, train_pred, epochs=7):
#     num_actors = 32
#
#     dataset = PostprocessingDataset(train_data, train_pred)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, shuffle=True, batch_size=num_actors, pin_memory=True, drop_last=True)
#
#     ppo = rl.ppo.PPO(
#         gym.spaces.Box(-1, 1, 1), gym.spaces.Box(-1, 1, 3),
#         num_actors=num_actors,
#         optimizer_factory=partial(torch.optim.SGD, lr=0.001, momentum=0.9),
#         policy_clip=0.2,
#         value_clip=0.2,
#         ppo_iters=1,
#         constraint='clip_mod',
#         grad_clip_norm=10,
#         horizon=1,
#         batch_size=num_actors,
#         learning_decay_frames=10e6,
#         model_factory=ConstActor,
#         image_observation=False,
#         cuda_eval=False,
#         cuda_train=False,
#         reward_discount=1,
#         advantage_discount=1,
#         value_loss_scale=0,
#         entropy_bonus=0.0,
#         reward_scale=1.0,
#     )
#
#     sys.stdout.flush()
#
#     for epoch in range(epochs):
#         with tqdm(dataloader) as pbar:
#             reward_ma, src_iou_ma, aug_iou_ma = 0, 0, 0
#             for i, batch in enumerate(pbar):
#                 img, pred_mask, pred_sdf, pred_cont, mask = [v.numpy() for v in batch]
#                 pred_msc = np.concatenate([pred_mask, pred_sdf, pred_cont], 1)
#                 pred_msc = pred_msc / (pred_msc.std((2, 3), keepdims=True) + 0.5)
#                 pred_unpad_mask = pred_mask[:, 0, pad:-pad, pad:-pad]
#                 src_iou = object_iou(pred_unpad_mask, mask.squeeze(1))
#
#                 # rl_input = np.concatenate([img, pred_msc], 1)
#                 # print(rl_input.shape)
#                 # rl_input = rl_input.reshape(rl_input.shape[0], -1)
#                 actions = ppo.eval(np.zeros((num_actors, 1)))
#                 # actions = actions.reshape(-1, size, size)
#                 sdf_unpad = pred_msc[:, 1, pad:-pad, pad:-pad]
#                 cont_unpad = pred_msc[:, 2, pad:-pad, pad:-pad]
#                 actions = [a.reshape(-1, 1, 1) for a in actions.reshape(-1, 3).T]
#                 rl_mask = pred_unpad_mask + actions[0] * sdf_unpad + actions[1] * cont_unpad + actions[2]
#
#                 aug_iou = object_iou(rl_mask, mask.squeeze(1))
#                 reward = (aug_iou - src_iou) * 5
#                 # print(actions.mean(), actions.std(), reward.mean(), reward.std())
#                 ppo.reward(reward)
#                 ppo.finish_episodes(np.ones(num_actors, dtype=np.bool))
#
#                 bc = 1 - 0.99 ** (i + 1)
#                 reward_ma = 0.99 * reward_ma + 0.01 * reward.mean()
#                 src_iou_ma = 0.99 * src_iou_ma + 0.01 * src_iou.mean()
#                 aug_iou_ma = 0.99 * aug_iou_ma + 0.01 * aug_iou.mean()
#                 pbar.set_postfix(epoch=epoch, src_iou=src_iou_ma / bc, aug_iou=aug_iou_ma / bc, r=reward_ma / bc, refresh=False)
#
#     return ppo.model.unet
#
#
# def train_postprocessor_es(train_data, train_pred, epochs=7):
#     es_samples = 8
#     batch_size = 6
#
#     dataset = PostprocessingDataset(train_data, train_pred)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True)
#
#     snes = SNES(np.array([0, 0, 0], dtype=np.float32), init_std=0.1, lr=0.01, std_step=0.00, pop_size=es_samples)
#
#     sys.stdout.flush()
#
#     for epoch in range(epochs):
#         with tqdm(dataloader) as pbar:
#             reward_ma, src_iou_ma, aug_iou_ma = 0, 0, 0
#             for i, batch in enumerate(pbar):
#                 img, pred_mask, pred_sdf, pred_cont, mask = [np.repeat(v.numpy(), es_samples, 0) for v in batch]
#                 mask = mask.squeeze(1)
#                 # pred_msc = np.concatenate([pred_mask, pred_sdf, pred_cont], 1)
#                 # pred_msc = pred_msc / (pred_msc.std((2, 3), keepdims=True) + 0.5)
#                 pred_unpad_mask = pred_mask[:, 0, pad:-pad, pad:-pad]
#
#                 src_iou = object_iou(pred_unpad_mask, mask)
#
#                 # rl_input = np.concatenate([img, pred_msc], 1)
#                 # # print(rl_input.shape)
#                 # rl_input = rl_input.reshape(rl_input.shape[0], -1)
#                 # actions = ppo.eval(rl_input)
#                 # actions = actions.reshape(-1, size, size)
#
#                 params = snes.get_batch_samples().reshape(3, es_samples, 1, 1)
#                 params = np.repeat(params, batch_size, 1)
#                 mod_mask = pred_unpad_mask + \
#                            pred_cont[:, 0, pad:-pad, pad:-pad] * params[0] + \
#                            pred_sdf[:, 0, pad:-pad, pad:-pad] * params[1] + \
#                            params[2]
#                 aug_iou = object_iou(mod_mask, mask)
#                 reward = (aug_iou - src_iou).reshape(es_samples, batch_size).mean(-1)
#                 snes.rate_batch_samples(reward)
#
#                 # snes.cur_solution = snes.cur_solution * 0.98 + snes.best_solution * 0.02
#
#                 # print(actions.mean(), actions.std(), reward.mean(), reward.std())
#                 # ppo.reward(reward)
#                 # ppo.finish_episodes(np.ones(num_actors, dtype=np.bool))
#
#                 bc = 1 - 0.99 ** (i + 1)
#                 reward_ma = 0.99 * reward_ma + 0.01 * reward.mean()
#                 src_iou_ma = 0.99 * src_iou_ma + 0.01 * src_iou.mean()
#                 aug_iou_ma = 0.99 * aug_iou_ma + 0.01 * aug_iou.mean()
#                 pbar.set_postfix(epoch=epoch, aug_iou=aug_iou_ma / bc, r=reward_ma / bc, c=snes.cur_solution, std=snes.std, refresh=False)
#
#     # return ppo.model.unet


def object_iou(pred_mask, target_mask, threshold=0.5):
    pmask = pred_mask.cpu().numpy().squeeze(1)
    tmask = target_mask.cpu().numpy().squeeze(1)
    aug_ious = [mean_threshold_object_iou(label(pm > threshold), tm) for pm, tm in zip(pmask, tmask)]
    return torch.Tensor(aug_ious).type_as(pred_mask)


def multilayer_object_iou(pred_mask_layer_probs, target_labels):
    tlabels = target_labels.cpu().numpy().squeeze(1)
    max_softmax_indices = pred_mask_layer_probs.max(1)[1].cpu().numpy()

    ious = []
    for (softmax_indexes_sample, target_labels_sample) in zip(max_softmax_indices, tlabels):
        softmax_labels = softmax_to_labels(softmax_indexes_sample, pred_mask_layer_probs.shape[1])
        mt_iou = mean_threshold_object_iou(softmax_labels, target_labels_sample)
        ious.append(mt_iou)

    return torch.Tensor(ious).type_as(pred_mask_layer_probs)


def softmax_to_labels(max_softmax_indices, softmax_size):
    label_layers = [label(max_softmax_indices == i) for i in range(1, softmax_size)]
    pmask = np.zeros_like(label_layers[0], dtype=np.int64)
    last_idx = 0
    for layer_label in label_layers:
        layer_count = layer_label.max()
        layer_label = layer_label.copy()
        layer_label[layer_label != 0] += last_idx
        assert not pmask[layer_label != 0].any()
        pmask += layer_label
        last_idx += layer_count
    return pmask


# def full_image_iou(pred_mask, mask):
#     pred = pred_mask > 0
#     target = mask > 0
#     union = (pred | target).sum((1, 2))
#     intersection = (pred & target).sum((1, 2))
#     iou = intersection / union.clip(min=1)
#     iou[union == 0] = 1
#     return iou


# class GanG(nn.Module):
#     def __init__(self, nf, num_channels, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(num_channels + 1, nf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(nf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(nf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(nf * 8),
#             nn.LeakyReLU(0.2),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(nf * 8, num_classes, 4, 1, 0, bias=False),
#         )
#
#     def forward(self, input):
#         noise = Variable(input.data.new(input.shape[0], 1, *input.shape[2:]).normal_(0, 1))
#         x = torch.cat([input, noise], 1)
#         classes = self.main(x).view(input.shape[0], self.num_classes)
#         return classes


class GanD_UNet(nn.Module):
    def __init__(self, nc=3, nf=64):
        super().__init__()
        self.unet = UNet(nc, 1, f=nf)

    def forward(self, input):
        features = self.unet(input)
        output = features.view(input.shape[0], -1).mean(-1)
        return output, features


class GanD(nn.Module):
    def __init__(self, num_channels, nf=64):
        super().__init__()
        self.start = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_channels, nf, 4, 2, 1, bias=False),
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
        self.end = nn.Sequential(
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image):
        features = self.start(image)
        output = self.end(features)
        output = output.view(output.shape[0])
        return output, features