import sys

import dill
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import optim
from optfn.cosine_annealing import CosineAnnealingRestartParam
from optfn.gadam import GAdam
from optfn.param_groups_getter import get_param_groups
from optfn.near_instance_norm import NearInstanceNorm2d
from optfn.batch_renormalization import BatchReNorm2d
from optfn.learned_norm import LearnedNorm2d
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import deque

from .feature_pyramid_network import FPN
from ..dataset import NucleiDataset
from ..iou import threshold_iou, iou
from ..roi_align import roi_align, pad_boxes
from ..settings import box_padding, train_pad, resnet_norm_mean, resnet_norm_std
from ..losses import soft_dice_loss
from optfn.gated_instance_norm import GatedInstanceNorm2d
from optfn.eval_batch_norm import EvalBatchNorm2d


def binary_cross_entropy_with_logits(x, z, reduce=True):
    bce = x.clamp(min=0) - x * z + x.abs().neg().exp().add(1).log()
    return bce.mean() if reduce else bce

# def binary_focal_loss_with_logits(x, t, gamma=2, alpha=0.25):
#     p = x.sigmoid()
#     pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
#     w = (1 - alpha) * t + alpha * (1 - t)  # w = 1-alpha if t > 0 else alpha
#     w = w * (1 - pt).pow(gamma)
#     return F.binary_cross_entropy_with_logits(x, t, w)


def binary_focal_loss_with_logits(input, target, lam=2):
    weight = (target - F.sigmoid(input)).abs().pow(lam)
    ce = weight * binary_cross_entropy_with_logits(input, target, reduce=False)
    return ce.mean()


# def mse_focal_loss(pred, target, focal_threshold, lam=2):
#     mse = F.mse_loss(pred, target, reduce=False)
#     w = (pred - target).clamp(-focal_threshold, focal_threshold).div(focal_threshold).abs().pow(lam).detach()
#     loss = w * mse
#     return loss.mean()


# def copy_state_dict(model):
#     return copy.deepcopy(OrderedDict((k, v.cpu()) for k, v in model.state_dict().items()))


def get_reg_loss(model):
    losses = []
    for module in model.modules():
        if hasattr(module, 'reg_loss') and module.reg_loss is not None:
            losses.append(module.reg_loss)
    return torch.cat(losses).sum() if len(losses) != 0 else 0


def adjust_norm_scheme(model):
    for module in model.modules():
        for name, old_norm in module.named_children():
            if isinstance(old_norm, nn.InstanceNorm2d):
                new_norm = GatedInstanceNorm2d(old_norm.num_features)
                # new_norm = nn.InstanceNorm2d(old_norm.num_features, affine=old_norm.affine)
                # new_norm = NearInstanceNorm2d(old_norm.num_features, affine=old_norm.affine)
                # if hasattr(new_norm, 'weight'):
                #     new_norm.weight = old_norm.weight
                # if hasattr(new_norm, 'bias'):
                #     new_norm.bias = old_norm.bias
                # if hasattr(new_norm, 'running_mean'):
                #     new_norm.running_mean = old_norm.running_mean
                # if hasattr(new_norm, 'running_var'):
                #     new_norm.running_var = old_norm.running_var
                # elif hasattr(new_norm, 'running_std'):
                #     new_norm.running_std = old_norm.running_var.sqrt_()
                setattr(module, name, new_norm)
            # if isinstance(old_norm, nn.BatchNorm2d):
            #     new_norm = EvalBatchNorm2d(old_norm.num_features, affine=old_norm.affine, eps=old_norm.eps)
            #     # new_norm = nn.InstanceNorm2d(old_norm.num_features, affine=old_norm.affine)
            #     # new_norm = NearInstanceNorm2d(old_norm.num_features, affine=old_norm.affine)
            #     if hasattr(new_norm, 'weight'):
            #         new_norm.weight = old_norm.weight
            #     if hasattr(new_norm, 'bias'):
            #         new_norm.bias = old_norm.bias
            #     if hasattr(new_norm, 'running_mean'):
            #         new_norm.running_mean = old_norm.running_mean
            #     if hasattr(new_norm, 'running_var'):
            #         new_norm.running_var = old_norm.running_var
            #     elif hasattr(new_norm, 'running_std'):
            #         new_norm.running_std = old_norm.running_var.sqrt_()
            #     setattr(module, name, new_norm)


class Trainer:
    def __init__(self):
        self.dataset = None
        self.dataloader = None
        self.model_gen = None
        self.optimizer_gen = None
        self.scheduler_gen = None
        self.prev_batch_time = None
        self.pbar = None
        self.summary: SummaryWriter = None
        self.optim_iter = 0
        self.scalar_history = dict()

    def train(self, train_data, epochs=15, pretrain_epochs=7, saved_model=None, model_save_path=None):
        self.dataset = NucleiDataset(train_data)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=1, pin_memory=True)

        self.model_gen = FPN(3) if saved_model is None else saved_model
        adjust_norm_scheme(self.model_gen)
        self.model_gen = self.model_gen.cuda()
        self.model_gen.freeze_pretrained_layers(False)

        self.optimizer_gen = GAdam(get_param_groups(self.model_gen), lr=0.03, weight_decay=5e-4)
        # self.optimizer_gen = optim.SGD(get_param_groups(self.model_gen), lr=0.02, momentum=0.9, weight_decay=1e-4)

        self.scheduler_gen = CosineAnnealingRestartParam(self.optimizer_gen, len(self.dataloader), 2)

        sys.stdout.flush()

        self.prev_batch_time = time.time()

        with SummaryWriter() as self.summary:
            self.summary.add_text('hparams', f'epochs {epochs}; pretrain {pretrain_epochs}')
            for epoch in range(epochs):
                with tqdm(self.dataloader) as self.pbar:
                    self.model_gen.freeze_pretrained_layers(epoch < pretrain_epochs)

                    for data in self.pbar:
                        self.optim_step(data)

                    if model_save_path is not None:
                        torch.save(self.model_gen, model_save_path, pickle_module=dill)

        return self.model_gen

    def optim_step(self, data):
        self.optimizer_gen.zero_grad()
        self.scheduler_gen.step()
        # scheduler_disc.step()

        img, labels, sdf = [x.cuda() for x in data]

        # img = (img - img.view(img.shape[0], -1).mean(-1).view(-1, 1, 1)).mul_(2)
        img += img.new(img.shape).normal_(0, 0.03)

        x_train = Variable(img)
        sdf_train = Variable(sdf * 0.5 + 0.5)
        mask_train = Variable((labels > 0).float())

        cont_train = 1 - (sdf.clamp(0.0, 0.25) * 8 - 1) ** 2
        cont_train = Variable(cont_train)

        model_out_layers, model_out_img = self.model_gen(x_train)
        unpad_slice = (Ellipsis, slice(train_pad, -train_pad), slice(train_pad, -train_pad))
        train_pairs = get_train_pairs(self.model_gen, labels, sdf, img, model_out_layers)
        img_mask_out, img_sdf_out, img_cont_out = model_out_img.split(1, 1)

        if train_pairs is None:
            self.optimizer_gen.zero_grad()
            return

        pred_features, target_masks, pred_scores, target_scores, \
        pred_boxes, target_boxes, pred_boxes_raw, target_boxes_raw, img_crops, layer_idx = train_pairs

        pred_masks = self.model_gen.predict_masks(pred_features)

        bm_idx, bm_count = np.unique(layer_idx, return_counts=True)
        batch_masks = np.zeros(len(self.model_gen.mask_pixel_sizes))
        batch_masks[bm_idx] += bm_count

        # if return_predictions_at_epoch is not None and return_predictions_at_epoch == epoch:
        #     return pred_masks.data.cpu(), target_masks.cpu(), img_crops.cpu(), \
        #            x_train.data.cpu(), labels.cpu(), model_out_img.data.cpu(), model_gen

        target_masks = Variable(target_masks)
        target_scores = Variable(target_scores)
        target_boxes_raw = Variable(target_boxes_raw)

        img_mask_loss = 0.1 * binary_focal_loss_with_logits(img_mask_out, mask_train.clamp(0.05, 0.95))
        img_cont_loss = 0.1 * binary_focal_loss_with_logits(img_cont_out, cont_train.clamp(0.05, 0.95))
        img_sdf_loss = 0.1 * binary_focal_loss_with_logits(img_sdf_out, sdf_train.clamp(0.05, 0.95))

        mask_loss = binary_focal_loss_with_logits(pred_masks, target_masks.clamp(0.05, 0.95))
        score_loss = binary_focal_loss_with_logits(pred_scores, target_scores.clamp(0.05, 0.95))
        box_loss = F.mse_loss(pred_boxes_raw, target_boxes_raw)
        reg_loss = 1e-3 * get_reg_loss(self.model_gen)

        loss = mask_loss + box_loss + score_loss + img_mask_loss + img_sdf_loss + img_cont_loss + reg_loss

        loss.backward()
        self.optimizer_gen.step()
        self.optimizer_gen.zero_grad()

        box_iou = aabb_iou(pred_boxes, target_boxes)

        pred_score_np = (pred_scores.data > 0).cpu().numpy().reshape(-1)
        target_score_np = (target_scores.data > 0.5).byte().cpu().numpy().reshape(-1)

        score_prec, score_rec, score_fscore, _ = precision_recall_fscore_support(
            pred_score_np, target_score_np, average='binary', warn_for=[])

        f_iou = iou(pred_masks.data > 0, target_masks.data > 0.5)
        t_iou = threshold_iou(f_iou)

        optim_iter = self.optim_iter + 1

        batch_time = time.time() - self.prev_batch_time
        self.prev_batch_time = time.time()
        if optim_iter >= 100:
            self.add_scalar_averaged('Batch Time', batch_time, optim_iter)

        if optim_iter >= 100 and optim_iter % 100 == 0:
            resnet_std = data[0].new(resnet_norm_std).view(-1, 1, 1)
            resnet_mean = data[0].new(resnet_norm_mean).view(-1, 1, 1)
            img_unnorm = data[0].mul(resnet_std).add_(resnet_mean)
            scores = model_out_layers[1][1][:, 3:6, train_pad // 8: -train_pad // 8, train_pad // 8: -train_pad // 8]
            good_looking_mask = data[1].float().mul(1 / (data[1].max() + 10)).clamp_(0, 1)
            self.summary.add_image('Train Image', img_unnorm[unpad_slice], optim_iter)
            self.summary.add_image('Train Mask', good_looking_mask[unpad_slice], optim_iter)
            self.summary.add_image('Train SDF', sdf_train.data[unpad_slice], optim_iter)
            self.summary.add_image('Train Contour', cont_train.data[unpad_slice], optim_iter)
            self.summary.add_image('Predicted Mask', torch.sigmoid(img_mask_out.data[unpad_slice]), optim_iter)
            self.summary.add_image('Predicted SDF', torch.sigmoid(img_sdf_out.data[unpad_slice]), optim_iter)
            self.summary.add_image('Predicted Contour', torch.sigmoid(img_cont_out.data[unpad_slice]), optim_iter)
            self.summary.add_image('Predicted Score', torch.sigmoid(scores), optim_iter)

        if optim_iter >= 100:
            self.add_scalar_averaged('Score FScore', score_fscore, optim_iter)
            self.add_scalar_averaged('Score Precision', score_prec, optim_iter)
            self.add_scalar_averaged('Score Recall', score_rec, optim_iter)
            self.add_scalar_averaged('Mask IoU Full', f_iou.mean(), optim_iter)
            self.add_scalar_averaged('Mask IoU Threshold', t_iou.mean(), optim_iter)
            self.add_scalar_averaged('Box IoU', box_iou.mean(), optim_iter)
            self.add_scalar_averaged('Learning Rate', self.optimizer_gen.param_groups[0]['lr'], optim_iter)
            self.summary.add_histogram('Masks Per Level', batch_masks, optim_iter)
            self.summary.add_histogram('Full Mask IoU Hist', f_iou, optim_iter)
            self.summary.add_histogram('Threshold Mask IoU Hist', t_iou, optim_iter)
            self.summary.add_histogram('Box IoU Hist', box_iou, optim_iter)

        self.optim_iter += 1

    def add_scalar_averaged(self, tag, scalar_value, global_step):
        hist_size = 50
        if tag not in self.scalar_history:
            self.scalar_history[tag] = deque(maxlen=hist_size)
        hist = self.scalar_history[tag]
        hist.append(scalar_value)
        if global_step % hist_size == 0:
            self.summary.add_scalar(tag, np.mean(hist), global_step)


def get_train_pairs(
        model, labels, sdf, img, net_out,
        pos_sdf_threshold=0.1, neg_sdf_threshold=-0.1,
        pos_iou_limit=0.4, neg_iou_limit=0.3,
        pos_samples=32, neg_to_pos_ratio=3):
    outputs = []
    for sample_idx in range(labels.shape[0]):
        net_out_sample = [(m[sample_idx], s[sample_idx], (b[0][sample_idx], b[1][sample_idx], b[2])) for m, s, b in net_out]
        o = get_train_pairs_single(
            model, labels[sample_idx, 0], sdf[sample_idx, 0], img[sample_idx], net_out_sample,
            model.mask_pixel_sizes, pos_sdf_threshold, neg_sdf_threshold,
            pos_iou_limit, neg_iou_limit,
            pos_samples, neg_to_pos_ratio
        )
        outputs.extend(o)

    if len(outputs) == 0:
        return None

    outputs = list(zip(*outputs))
    outputs, layer_idx = outputs[:-1], np.concatenate(outputs[-1])
    pred_features, target_masks, pred_scores, target_scores, \
    pred_boxes, target_boxes, pred_boxes_raw, target_boxes_raw, img_crops = \
        [torch.cat(o, 0) for o in outputs]
    return pred_features, target_masks, pred_scores, target_scores, \
           pred_boxes, target_boxes, pred_boxes_raw, target_boxes_raw, img_crops, layer_idx


def get_train_pairs_single(model, labels, sdf, img, net_out, pixel_sizes,
                           pos_sdf_threshold, neg_sdf_threshold,
                           pos_iou_limit, neg_iou_limit,
                           pos_samples, neg_to_pos_ratio):
    box_mask = get_object_boxes(labels, 2)
    resampled_layers = resample_data(labels, sdf, box_mask, img, pixel_sizes)
    outputs = []
    for layer_idx, layer_data in enumerate(zip(net_out, resampled_layers)):
        (out_masks, out_scores, out_boxes), (res_labels, res_sdf, res_boxes, res_img) = layer_data

        o = generate_samples_for_layer(
            model, out_masks, out_scores, out_boxes, res_labels, res_sdf, res_boxes, res_img,
            pos_sdf_threshold, neg_sdf_threshold,
            pos_iou_limit, neg_iou_limit,
            pos_samples, neg_to_pos_ratio
        )
        if o is not None:
            outputs.append((*o, o[0].shape[0] * [layer_idx]))
    return outputs


def generate_samples_for_layer(model, out_features, out_scores, out_boxes, labels, sdf, obj_boxes, img,
                               pos_sdf_threshold, neg_sdf_threshold,
                               pos_iou_limit, neg_iou_limit,
                               pos_samples, neg_to_pos_ratio, box_padding=box_padding):
    out_boxes, raw_boxes, anchor_boxes = out_boxes

    # border and stride for converting between image space and conv-center space
    stride = model.mask_size // model.region_size
    border = stride // 2

    # slice to select values from image at conv center locations
    mask_centers_slice = (
        slice(border, -border + 1, stride),
        slice(border, -border + 1, stride))
    # [fs, fs] - sdf at conv centers
    sdf_fs = sdf[mask_centers_slice]
    # [(y, x, h, w), fs, fs] - obj boxes at conv centers
    target_boxes_fs = obj_boxes[(slice(None), *mask_centers_slice)].contiguous()
    # target_boxes_fs[0] -= target_boxes_fs[2] * box_padding
    # target_boxes_fs[1] -= target_boxes_fs[3] * box_padding
    # target_boxes_fs[2:] *= 1 + 2 * box_padding
    target_boxes_fs = target_boxes_fs.float() / target_boxes_fs.new(2 * [*labels.shape[:2]]).view(-1, 1, 1)

    assert target_boxes_fs.shape[-1] == anchor_boxes.shape[-1], \
        (target_boxes_fs.shape, anchor_boxes.shape, out_boxes.shape, out_features.shape,
         labels.shape, sdf.shape, img.shape)

    anchor_iou = aabb_iou(
        target_boxes_fs.unsqueeze(1).expand_as(anchor_boxes).contiguous().view(anchor_boxes.shape[0], -1).t(),
        anchor_boxes.view(anchor_boxes.shape[0], -1).t()
    ).view(*anchor_boxes.shape[1:])

    # assert sdf_fs.shape == out_masks.shape[-2:], (sdf_fs.shape, out_masks.shape)

    pos_centers_fmap = (anchor_iou > pos_iou_limit) & (sdf_fs > pos_sdf_threshold)
    neg_centers_fmap = (anchor_iou < neg_iou_limit) | (sdf_fs < neg_sdf_threshold)

    # TODO: allow zero negative centers
    if pos_centers_fmap.sum() == 0 or neg_centers_fmap.sum() == 0:
        return None

    pos_centers_fmap_idx_all = pos_centers_fmap.view(-1).nonzero().squeeze()
    neg_centers_fmap_idx_all = neg_centers_fmap.view(-1).nonzero().squeeze()
    pos_centers_fmap_perm = torch.randperm(len(pos_centers_fmap_idx_all))
    # neg_centers_fmap_perm = torch.randperm(len(neg_centers_fmap_idx_all))
    pos_centers_fmap_perm = pos_centers_fmap_perm[:pos_samples].contiguous().cuda()
    # neg_centers_fmap_perm = neg_centers_fmap_perm[:len(pos_centers_fmap_perm) * neg_to_pos_ratio].contiguous().cuda()
    pos_centers_fmap_idx = pos_centers_fmap_idx_all[pos_centers_fmap_perm]
    # neg_centers_fmap_idx = neg_centers_fmap_idx_all[neg_centers_fmap_perm]

    pred_pos_scores = out_scores.take(Variable(pos_centers_fmap_idx_all))
    pred_neg_scores = out_scores.take(Variable(neg_centers_fmap_idx_all))
    pred_scores = torch.cat([pred_pos_scores, pred_neg_scores])
    target_scores = out_scores.data.new(pred_scores.shape[0]).fill_(0)
    target_scores[:pred_pos_scores.shape[0]] = 1

    # ([y, x, h, w], NPos)
    mask_boxes = out_boxes.data.view(out_boxes.shape[0], -1)[:, pos_centers_fmap_idx]
    pred_boxes_raw = raw_boxes.view(raw_boxes.shape[0], -1)[:, pos_centers_fmap_idx_all]
    pred_boxes = out_boxes.data.view(out_boxes.shape[0], -1)[:, pos_centers_fmap_idx_all]
    target_boxes = target_boxes_fs.unsqueeze(1).repeat(1, anchor_boxes.shape[1], 1, 1)
    target_boxes = target_boxes.view(target_boxes_fs.shape[0], -1)[:, pos_centers_fmap_idx_all]
    anchor_boxes_selected = anchor_boxes.view(target_boxes_fs.shape[0], -1)[:, pos_centers_fmap_idx_all]
    target_boxes_raw = to_raw_boxes(target_boxes, anchor_boxes_selected, anchor_boxes.shape[-2:])

    mask_boxes = mask_boxes.t()
    pred_boxes_raw, target_boxes_raw = pred_boxes_raw.t(), target_boxes_raw.t()
    pred_boxes, target_boxes = pred_boxes.t(), target_boxes.t()

    mask_boxes = pad_boxes(mask_boxes, box_padding)

    pred_features = roi_align(out_features.unsqueeze(0), mask_boxes, model.region_size)
    img_crops = roi_align(
        Variable(img.unsqueeze(0), volatile=True),
        mask_boxes,
        model.mask_size
    ).data

    pos_center_label_nums = labels[mask_centers_slice].unsqueeze(0).repeat(anchor_boxes.shape[1], 1, 1)
    pos_center_label_nums = pos_center_label_nums.take(pos_centers_fmap_idx)

    target_masks = labels_to_mask_roi_align(labels, mask_boxes, pos_center_label_nums, model.mask_size)

    return pred_features, target_masks, pred_scores, target_scores, \
           pred_boxes, target_boxes, pred_boxes_raw, target_boxes_raw, img_crops


def to_raw_boxes(boxes, anchor_boxes, fmap_shape):
    assert boxes.shape == anchor_boxes.shape
    assert boxes.dim() == 2
    assert boxes.shape[0] == 4
    assert len(fmap_shape) == 2

    boxes, anchor_boxes = boxes.clone(), anchor_boxes.clone()

    hwhw = boxes.new(2 * [*fmap_shape]).unsqueeze(1)
    boxes /= hwhw
    anchor_boxes /= hwhw

    anchor_boxes[:2] += anchor_boxes[2:] / 2

    def logit(x):
        return (x / (1 - x)).log_()

    boxes[:2].add_(boxes[2:] / 2).sub_(anchor_boxes[:2]).div_(anchor_boxes[2:])
    boxes[2:].div_(anchor_boxes[2:]).log_()
    boxes = logit(boxes.add_(1).div_(2))
    return boxes


def get_object_boxes(labels, downsampling=1):
    # [size, size]
    assert labels.dim() == 2
    assert labels.shape[0] == labels.shape[1]

    src_labels_shape = labels.shape
    if downsampling != 1:
        labels = downscale_nonzero(labels, downsampling)

    count = labels.max()
    if count == 0:
        return torch.zeros(4, *src_labels_shape).cuda()

    # [count] with [1, count]
    label_nums = torch.arange(1, count + 1).long().cuda()
    # [count, size, size] with [0, 1]
    masks = (labels.unsqueeze(0) == label_nums.view(-1, 1, 1)).float()

    # [new count] with [1, old count]
    nonzero_idx = (masks.view(masks.shape[0], -1).sum(-1) != 0).nonzero().squeeze()
    count = len(nonzero_idx)
    assert count != 0
    # [count, size, size] with [0, 1]
    masks = masks.index_select(0, nonzero_idx)

    # [size] with [0, size - 1] ascending
    size_range = torch.arange(labels.shape[0]).cuda()
    # [size] with [0, size - 1] descending
    size_range_rev = torch.arange(labels.shape[0] - 1, -1, -1).cuda()

    # [count, size, size] with [0, size), filtered by mask, ascending by 1 dim
    y_range_mask = masks * size_range.view(1, -1, 1)
    # [count, size, size] with [0, size), filtered by mask, descending by 1 dim
    y_range_mask_rev = masks * size_range_rev.view(1, -1, 1)
    # [count, size, size] with [0, size), filtered by mask, ascending by 2 dim
    x_range_mask = masks * size_range.view(1, 1, -1)
    # [count, size, size] with [0, size), filtered by mask, descending by 2 dim
    x_range_mask_rev = masks * size_range_rev.view(1, 1, -1)

    # [count] with [0, size)
    y_max = y_range_mask.view(count, -1).max(1)[0]
    x_max = x_range_mask.view(count, -1).max(1)[0]
    y_min = labels.shape[0] - 1 - y_range_mask_rev.view(count, -1).max(1)[0]
    x_min = labels.shape[0] - 1 - x_range_mask_rev.view(count, -1).max(1)[0]
    assert y_min.dim() == 1, y_min.shape

    # [count, 4] with [0, size], in format [y, x, h, w] at dim 1
    box_vec = torch.stack([y_min, x_min, y_max - y_min, x_max - x_min], 1)
    assert box_vec.shape == (count, 4)
    # [count, 4, size, size], filtered by mask, in format [y, x, h, w] at dim 1
    box_mask = masks.unsqueeze(1) * box_vec.view(count, 4, 1, 1)
    # [4, size, size], filtered by mask, in format [y, x, h, w] at dim 0
    box_mask = box_mask.sum(0)

    if downsampling != 1:
        box_mask = F.upsample(box_mask.unsqueeze(0), scale_factor=downsampling, mode='nearest').data.squeeze(0)
        box_mask *= downsampling

    return box_mask


def resample_data(labels, sdf, box_mask, img, pixel_sizes):
    # labels - [size, size]
    # sdf - [size, size]
    # box_mask - [4, size, size]
    # img - [3, size, size]
    # pixel_sizes - [layers count]
    assert labels.shape == sdf.shape
    assert labels.dim() == 2
    resampled = []
    for px_size in pixel_sizes:
        assert labels.shape[-1] % px_size == 0
        if px_size == 1:
            res_labels, res_sdf, res_boxes, res_img = labels, sdf, box_mask, img
        elif px_size < 1:
            assert round(1 / px_size, 3) == int(1 / px_size)
            factor = int(1 / px_size)
            res_labels = F.upsample(labels.view(1, 1, *labels.shape).float(), scale_factor=factor).data[0, 0].long()
            res_boxes = F.upsample(box_mask.unsqueeze(0), scale_factor=factor).data[0] / px_size
            res_sdf = F.upsample(sdf.view(1, 1, *sdf.shape), scale_factor=factor, mode='bilinear').data[0, 0]
            res_img = F.upsample(img.unsqueeze(0), scale_factor=factor, mode='bilinear').data[0]
        else:
            res_labels = downscale_nonzero(labels, px_size)
            res_boxes = downscale_nonzero(box_mask, px_size) / px_size
            res_sdf = F.avg_pool2d(Variable(sdf.view(1, 1, *sdf.shape), volatile=True), px_size, px_size).data[0, 0]
            res_img = F.avg_pool2d(Variable(img.unsqueeze(0), volatile=True), px_size, px_size).data[0]
        resampled.append((res_labels, res_sdf, res_boxes, res_img))
    return resampled


def downscale_nonzero(x, factor):
    assert x.shape[-1] == x.shape[-2]
    assert x.shape[-1] % factor == 0
    assert 2 <= x.dim() <= 4
    ns = x.shape[-1]
    ds = x.shape[-1] // factor
    # [norm size, norm size] - non zero column / pixel mask
    mask = x.view(-1, ns, ns).sum(0) != 0
    # [down size, down size, factor, factor]
    mask = mask.view(ds, factor, ds, factor).transpose(1, 2).float()
    zero_mask = (mask.view(ds, ds, -1).mean(-1) < 0.5).view(ds, ds, 1, 1).expand_as(mask)
    # inverting values of mask cells where max() should select zeros
    mask[zero_mask] *= -1
    # [-1, ds, ds, factor, factor]
    gx = x.view(-1, ds, factor, ds, factor).transpose(2, 3).contiguous()
    # [-1, ds, ds, factor * factor]
    gx = gx.view(*gx.shape[:3], -1)
    # [ds, ds] indices
    downscaled_indices = mask.view(ds, ds, -1).max(-1)[1]
    # [gx[0], ds, ds, 1]
    downscaled_indices = downscaled_indices.view(1, ds, ds, 1).expand(gx.shape[0], ds, ds, 1)
    gx = gx.gather(-1, downscaled_indices)
    gx = gx.sum(-1)
    gx = gx.view(*x.shape[:-2], ds, ds)
    assert gx.shape == (*x.shape[:-2], ds, ds), (x.shape, gx.shape)
    return gx


def labels_to_mask_roi_align(labels, boxes, label_nums, crop_size):
    assert labels.dim() == 2
    assert boxes.dim() == 2
    assert label_nums.dim() == 1
    masks = []
    for box, label_num in zip(boxes, label_nums):
        mask = (labels == label_num).view(1, 1, *labels.shape).float()
        mask = roi_align(Variable(mask, volatile=True), box.unsqueeze(0), crop_size).data
        masks.append(mask)
    return torch.cat(masks, 0)


def aabb_iou(a, b):
    assert a.dim() == 2 and a.shape[1] == 4
    assert a.shape == b.shape

    a, b = a.t(), b.t()

    inter_top = torch.max(a[0], b[0])
    inter_left = torch.max(a[1], b[1])
    inter_bot = torch.min(a[0] + a[2], b[0] + b[2])
    inter_right = torch.min(a[1] + a[3], b[1] + b[3])

    intersection = (inter_right - inter_left) * (inter_bot - inter_top)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    iou = intersection / (area_a + area_b - intersection)
    iou[(inter_right < inter_left) | (inter_bot < inter_top)] = 0
    return iou


# def center_crop(image, centers, border, centers_img_size):
#     """
#     Make several crops of image
#     Args:
#         image: Cropped image. Can have any nuber of channels, only last two are used for cropping.
#         centers: 1d indexes of crop centers.
#         border: tuple of (left-top, right-bottom) offsets from `centers`.
#         centers_img_size: Size of image used to convert centers from 1d to 2d format.
#
#     Returns: Tensor with crops [num crops, ..., crop size, crop size]
#
#     """
#     # get 2d indexes of `centers`
#     centers_y = centers / centers_img_size
#     centers_x = centers - centers_y * centers_img_size
#     centers = torch.stack([centers_y, centers_x], 1).cpu()
#     assert centers.shape == (centers_x.shape[0], 2), centers.shape
#     # crop `image` in +-border range from centers
#     crops = []
#     for c in centers:
#         crop = image[..., c[0] + border[0]: c[0] + border[1], c[1] + border[0]: c[1] + border[1]]
#         crops.append(crop)
#     return torch.stack(crops, 0)


# def mask_to_indexes(mask, stride, border, size):
#     """
#     Convert binary mask to indexes and upscale them
#     Args:
#         mask: Binary mask
#         stride: Stride between mask cells
#         border: Mask padding
#         size: Size of upscaled image
#
#     Returns: 1d indexes
#
#     """
#     # convert `mask` from binary mask to 2d indexes and upscale them from conv-center space to image space
#     idx = mask.nonzero() * stride + border
#     # convert to flat indexes
#     return idx[:, 0] * size + idx[:, 1]


# # custom weights initialization called on netG and netD
# def weights_init(m):
#     if isinstance(m, _ConvNd) or isinstance(m, nn.Linear):
#         m.weight.data.normal_(0.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, _BatchNorm):
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# def calc_gradient_penalty(netD, real_data, fake_data):
#     LAMBDA = 2
#     alpha = torch.rand(real_data.shape[0], 1, 1, 1)
#     # alpha = alpha.expand(real_data.shape)
#     alpha = alpha.cuda()
#
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#     interpolates = interpolates.cuda()
#     interpolates = Variable(interpolates, requires_grad=True)
#
#     disc_interpolates, _ = netD(interpolates)
#
#     gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#     return gradient_penalty
#
#
# class GanD(nn.Module):
#     def __init__(self, nc=3, nf=128):
#         super().__init__()
#         self.head = nn.Sequential(
#             # input is (nc) x 32 x 32
#             nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
#             nn.ReLU(inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(nf * 2),
#             nn.ReLU(inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
#         )
#         self.tail = nn.Sequential(
#             nn.BatchNorm2d(nf * 4),
#             nn.ReLU(inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(nf * 4, 1, 4, 1, 0, bias=False),
#         )
#
#     def forward(self, input):
#         features = self.head(input)
#         output = self.tail(features)
#         return output.view(-1), features