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
from scipy.ndimage import gaussian_filter

from .dataset import make_train_dataset
from .dataset import pad, size
from .iou import mean_threshold_object_iou
from .losses import dice_loss
from .transforms import RandomCrop, Pad
from .unet import UNet
from .dataset import resnet_norm_mean, resnet_norm_std
from .postprocessing_dataset import PostprocessingDataset
from optfn.gadam import GAdam
import gym.spaces
from functools import partial
from .unet_actor import UNetActorCritic
from skimage.morphology import label
from optfn.snes import SNES
from meta_rl.const_actor import ConstActor
from .unet_actor import SimpleCNNActor

try:
    import ppo_pytorch.ppo_pytorch as rl
except:
    import ppo_pytorch as rl


def train_postprocessor_ppo(train_data, train_pred, epochs=7):
    num_actors = 16

    dataset = PostprocessingDataset(train_data, train_pred)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=num_actors, pin_memory=True, drop_last=True)

    ppo = rl.ppo.PPO(
        gym.spaces.Box(-1, 1, 3 * (size + pad * 2) ** 2), gym.spaces.Box(-1, 1, 128 * 128),
        num_actors=num_actors,
        optimizer_factory=partial(GAdam, lr=0.0001, avg_sq_mode='weight', amsgrad=True, weight_decay=1e-4),
        policy_clip=0.2,
        value_clip=0.001,
        ppo_iters=1,
        constraint='clip_mod',
        grad_clip_norm=2,
        horizon=1,
        batch_size=4,
        model_factory=SimpleCNNActor,
        image_observation=False,
        cuda_eval=True,
        cuda_train=True,
        reward_discount=1,
        advantage_discount=1,
        value_loss_scale=0.1,
        entropy_bonus=1e-4,
        reward_scale=1.0,
        lr_scheduler_factory=None,
        clip_decay_factory=None,
        entropy_decay_factory=None,
    )

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            reward_ma, src_iou_ma, aug_iou_ma = 0, 0, 0
            for i, batch in enumerate(pbar):
                img, pred_mask, pred_sdf, pred_cont, mask = [v.numpy() for v in batch]
                pred_msc = np.concatenate([pred_mask, pred_sdf, pred_cont], 1)
                pred_msc = pred_msc / (pred_msc.std((2, 3), keepdims=True) + 0.5)
                pred_unpad_mask = pred_mask[:, 0, pad:-pad, pad:-pad]

                src_iou = object_iou(pred_unpad_mask, mask.squeeze(1))

                rl_input = pred_msc # np.concatenate([img, pred_msc], 1)
                rl_input = rl_input.reshape(rl_input.shape[0], -1)
                actions = ppo.eval(rl_input)
                actions = actions.reshape(-1, 1, size, size)
                actions = Variable(torch.from_numpy(actions), volatile=True)
                actions = F.avg_pool2d(actions, kernel_size=17, stride=1, padding=8)
                actions = actions.data.numpy().reshape(-1, size, size)

                rl_mask = pred_unpad_mask + actions

                # rl_input = pred_msc # np.concatenate([img, pred_msc], 1)
                # rl_input = rl_input.reshape(rl_input.shape[0], -1)
                # actions = actions.reshape(-1, size, size)
                # sdf_unpad = pred_msc[:, 1, pad:-pad, pad:-pad]
                # cont_unpad = pred_msc[:, 2, pad:-pad, pad:-pad]
                # actions = [a.reshape(-1, 1, 1) for a in actions.reshape(-1, 3).T]
                # rl_mask = pred_unpad_mask + actions[0] * sdf_unpad + actions[1] * cont_unpad + actions[2]

                aug_iou = object_iou(rl_mask, mask.squeeze(1))
                reward = (aug_iou - src_iou) * 10
                # print(actions.mean(), actions.std(), reward.mean(), reward.std())
                ppo.reward(reward)
                ppo.finish_episodes(np.ones(num_actors, dtype=np.bool))

                bc = 1 - 0.99 ** (i + 1)
                reward_ma = 0.99 * reward_ma + 0.01 * reward.mean()
                src_iou_ma = 0.99 * src_iou_ma + 0.01 * src_iou.mean()
                aug_iou_ma = 0.99 * aug_iou_ma + 0.01 * aug_iou.mean()
                pbar.set_postfix(epoch=epoch, src_iou=src_iou_ma / bc, aug_iou=aug_iou_ma / bc, r=reward_ma / bc, refresh=False)

    return ppo.model.unet


def train_postprocessor_ppo_const(train_data, train_pred, epochs=7):
    num_actors = 32

    dataset = PostprocessingDataset(train_data, train_pred)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=num_actors, pin_memory=True, drop_last=True)

    ppo = rl.ppo.PPO(
        gym.spaces.Box(-1, 1, 1), gym.spaces.Box(-1, 1, 3),
        num_actors=num_actors,
        optimizer_factory=partial(torch.optim.SGD, lr=0.001, momentum=0.9),
        policy_clip=0.2,
        value_clip=0.2,
        ppo_iters=1,
        constraint='clip_mod',
        grad_clip_norm=10,
        horizon=1,
        batch_size=num_actors,
        learning_decay_frames=10e6,
        model_factory=ConstActor,
        image_observation=False,
        cuda_eval=False,
        cuda_train=False,
        reward_discount=1,
        advantage_discount=1,
        value_loss_scale=0,
        entropy_bonus=0.0,
        reward_scale=1.0,
    )

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            reward_ma, src_iou_ma, aug_iou_ma = 0, 0, 0
            for i, batch in enumerate(pbar):
                img, pred_mask, pred_sdf, pred_cont, mask = [v.numpy() for v in batch]
                pred_msc = np.concatenate([pred_mask, pred_sdf, pred_cont], 1)
                pred_msc = pred_msc / (pred_msc.std((2, 3), keepdims=True) + 0.5)
                pred_unpad_mask = pred_mask[:, 0, pad:-pad, pad:-pad]
                src_iou = object_iou(pred_unpad_mask, mask.squeeze(1))

                # rl_input = np.concatenate([img, pred_msc], 1)
                # print(rl_input.shape)
                # rl_input = rl_input.reshape(rl_input.shape[0], -1)
                actions = ppo.eval(np.zeros((num_actors, 1)))
                # actions = actions.reshape(-1, size, size)
                sdf_unpad = pred_msc[:, 1, pad:-pad, pad:-pad]
                cont_unpad = pred_msc[:, 2, pad:-pad, pad:-pad]
                actions = [a.reshape(-1, 1, 1) for a in actions.reshape(-1, 3).T]
                rl_mask = pred_unpad_mask + actions[0] * sdf_unpad + actions[1] * cont_unpad + actions[2]

                aug_iou = object_iou(rl_mask, mask.squeeze(1))
                reward = (aug_iou - src_iou) * 5
                # print(actions.mean(), actions.std(), reward.mean(), reward.std())
                ppo.reward(reward)
                ppo.finish_episodes(np.ones(num_actors, dtype=np.bool))

                bc = 1 - 0.99 ** (i + 1)
                reward_ma = 0.99 * reward_ma + 0.01 * reward.mean()
                src_iou_ma = 0.99 * src_iou_ma + 0.01 * src_iou.mean()
                aug_iou_ma = 0.99 * aug_iou_ma + 0.01 * aug_iou.mean()
                pbar.set_postfix(epoch=epoch, src_iou=src_iou_ma / bc, aug_iou=aug_iou_ma / bc, r=reward_ma / bc, refresh=False)

    return ppo.model.unet


def train_postprocessor_es(train_data, train_pred, epochs=7):
    es_samples = 8
    batch_size = 6

    dataset = PostprocessingDataset(train_data, train_pred)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True)

    snes = SNES(np.array([0, 0, 0], dtype=np.float32), init_std=0.1, lr=0.01, std_step=0.00, pop_size=es_samples)

    sys.stdout.flush()

    for epoch in range(epochs):
        with tqdm(dataloader) as pbar:
            reward_ma, src_iou_ma, aug_iou_ma = 0, 0, 0
            for i, batch in enumerate(pbar):
                img, pred_mask, pred_sdf, pred_cont, mask = [np.repeat(v.numpy(), es_samples, 0) for v in batch]
                mask = mask.squeeze(1)
                # pred_msc = np.concatenate([pred_mask, pred_sdf, pred_cont], 1)
                # pred_msc = pred_msc / (pred_msc.std((2, 3), keepdims=True) + 0.5)
                pred_unpad_mask = pred_mask[:, 0, pad:-pad, pad:-pad]

                src_iou = object_iou(pred_unpad_mask, mask)

                # rl_input = np.concatenate([img, pred_msc], 1)
                # # print(rl_input.shape)
                # rl_input = rl_input.reshape(rl_input.shape[0], -1)
                # actions = ppo.eval(rl_input)
                # actions = actions.reshape(-1, size, size)

                params = snes.get_batch_samples().reshape(3, es_samples, 1, 1)
                params = np.repeat(params, batch_size, 1)
                mod_mask = pred_unpad_mask + \
                           pred_cont[:, 0, pad:-pad, pad:-pad] * params[0] + \
                           pred_sdf[:, 0, pad:-pad, pad:-pad] * params[1] + \
                           params[2]
                aug_iou = object_iou(mod_mask, mask)
                reward = (aug_iou - src_iou).reshape(es_samples, batch_size).mean(-1)
                snes.rate_batch_samples(reward)

                # snes.cur_solution = snes.cur_solution * 0.98 + snes.best_solution * 0.02

                # print(actions.mean(), actions.std(), reward.mean(), reward.std())
                # ppo.reward(reward)
                # ppo.finish_episodes(np.ones(num_actors, dtype=np.bool))

                bc = 1 - 0.99 ** (i + 1)
                reward_ma = 0.99 * reward_ma + 0.01 * reward.mean()
                src_iou_ma = 0.99 * src_iou_ma + 0.01 * src_iou.mean()
                aug_iou_ma = 0.99 * aug_iou_ma + 0.01 * aug_iou.mean()
                pbar.set_postfix(epoch=epoch, aug_iou=aug_iou_ma / bc, r=reward_ma / bc, c=snes.cur_solution, std=snes.std, refresh=False)

    # return ppo.model.unet


def object_iou(pred_mask, mask):
    aug_ious = []
    for pm, tm in zip(pred_mask, mask):
        iou = mean_threshold_object_iou(label(pm > 0), tm)
        aug_ious.append(iou)
    return np.array(aug_ious)


def full_image_iou(pred_mask, mask):
    pred = pred_mask > 0
    target = mask > 0
    union = (pred | target).sum((1, 2))
    intersection = (pred & target).sum((1, 2))
    iou = intersection / union.clip(min=1)
    iou[union == 0] = 1
    return iou
