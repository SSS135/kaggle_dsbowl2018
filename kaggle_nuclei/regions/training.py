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
import math
from ..rotated_box_intersection import intersection_area
from torchvision.utils import make_grid

from .feature_pyramid_network import FPN
from ..dataset import NucleiDataset
from ..iou import threshold_iou, iou
from ..roi_align import roi_align
from ..settings import box_padding, train_pad, resnet_norm_mean, resnet_norm_std
from ..losses import soft_dice_loss_with_logits
from optfn.gated_instance_norm import GatedInstanceNorm2d
from optfn.eval_batch_norm import EvalBatchNorm2d
from ..grad_running_norm import GradRunningNorm


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
        self.grn_full_mask = GradRunningNorm(0.1).cuda()
        self.grn_full_contour = GradRunningNorm(0.1).cuda()
        self.grn_full_sdf = GradRunningNorm(0.1).cuda()
        self.grn_patch_mask = GradRunningNorm().cuda()
        self.grn_score = GradRunningNorm().cuda()
        self.grn_box_mean = GradRunningNorm().cuda()
        self.grn_box_cov = GradRunningNorm().cuda()

    def train(self, train_data, epochs=15, pretrain_epochs=7, saved_model=None, model_save_path=None):
        self.dataset = NucleiDataset(train_data)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=1, pin_memory=True)

        self.model_gen = FPN(3) if saved_model is None else saved_model
        adjust_norm_scheme(self.model_gen)
        self.model_gen = self.model_gen.cuda()
        self.model_gen.freeze_pretrained_layers(False)

        self.optimizer_gen = GAdam(get_param_groups(self.model_gen), lr=3e-4, weight_decay=1e-4,
                                   nesterov=0.5, avg_sq_mode='output')
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
        loss = self.calc_loss(data)
        if loss is not None:
            loss.backward()
            self.optimizer_gen.step()
        self.optimizer_gen.zero_grad()
        self.scheduler_gen.step()

    def calc_loss(self, data):
        img, labels, sdf = [x.cuda() for x in data]

        # img = (img - img.view(img.shape[0], -1).mean(-1).view(-1, 1, 1)).mul_(2)
        img += img.new(img.shape).normal_(0, 0.03)

        x_train = Variable(img)
        sdf_train = Variable(sdf * 0.5 + 0.5)
        mask_train = Variable((labels > 0).float())

        cont_train = 1 - (sdf.clamp(0.0, 0.25) * 8 - 1) ** 2
        cont_train = Variable(cont_train)

        model_out_layers, model_out_img = self.model_gen(x_train)
        train_pairs = get_train_pairs(self.model_gen, labels, sdf, img, model_out_layers)
        img_mask_out, img_sdf_out, img_cont_out = model_out_img.split(1, 1)

        if train_pairs is None:
            self.optimizer_gen.zero_grad()
            return None

        pred_features, target_masks, pred_scores, target_scores, \
        pred_boxes, target_boxes, img_crops, true_masks, layer_idx = train_pairs

        pred_masks = self.model_gen.predict_masks(pred_features)

        bm_idx, bm_count = np.unique(layer_idx, return_counts=True)
        batch_masks = np.zeros(len(self.model_gen.mask_pixel_sizes))
        batch_masks[bm_idx] += bm_count

        target_masks = Variable(target_masks)
        target_scores = Variable(target_scores)
        target_boxes = Variable(target_boxes)

        img_mask_loss = soft_dice_loss_with_logits(self.grn_full_mask(img_mask_out), mask_train)
        img_cont_loss = binary_focal_loss_with_logits(self.grn_full_contour(img_cont_out), cont_train.clamp(0.05, 0.95))
        img_sdf_loss = binary_focal_loss_with_logits(self.grn_full_sdf(img_sdf_out), sdf_train.clamp(0.05, 0.95))

        mask_loss = soft_dice_loss_with_logits(self.grn_patch_mask(pred_masks), target_masks)
        score_loss = binary_focal_loss_with_logits(self.grn_score(pred_scores), target_scores.clamp(0.05, 0.95))
        box_loss = F.mse_loss(self.grn_box_mean(pred_boxes[:, :2]), target_boxes[:, :2]) + \
                   F.l1_loss(self.grn_box_cov(pred_boxes[:, 2:]), target_boxes[:, 2:])
        reg_loss = 1e-3 * get_reg_loss(self.model_gen)

        loss = mask_loss + box_loss + score_loss + img_mask_loss + img_sdf_loss + img_cont_loss + reg_loss

        box_iou = rotated_box_iou(pred_boxes.data, target_boxes.data)

        pred_score_np = (pred_scores.data > 0).cpu().numpy().reshape(-1)
        target_score_np = (target_scores.data > 0.5).byte().cpu().numpy().reshape(-1)

        score_prec, score_rec, score_fscore, _ = precision_recall_fscore_support(
            pred_score_np, target_score_np, average='binary', warn_for=[])

        f_iou = iou(pred_masks.data > 0, target_masks.data > 0.5)
        t_iou = threshold_iou(f_iou)

        optim_iter = self.optim_iter + 1

        batch_time = time.time() - self.prev_batch_time
        self.prev_batch_time = time.time()

        if optim_iter >= 100 and optim_iter % 100 == 0:
            unpad_slice = (Ellipsis, slice(train_pad, -train_pad), slice(train_pad, -train_pad))
            resnet_std = data[0].new(resnet_norm_std).view(-1, 1, 1)
            resnet_mean = data[0].new(resnet_norm_mean).view(-1, 1, 1)
            img_unnorm = data[0].mul(resnet_std).add_(resnet_mean)
            good_looking_mask = data[1].float().mul(1 / (data[1].max() + 10)).clamp_(0, 1)

            up_shape = model_out_layers[0][1].shape[2:]
            scores = torch.cat([F.upsample(l[1], size=up_shape, mode='bilinear') for l in model_out_layers], 1)
            scores = scores[..., train_pad // 4: -train_pad // 4, train_pad // 4: -train_pad // 4]

            self.summary.add_image('Train Image', img_unnorm[unpad_slice], optim_iter)
            self.summary.add_image('Train Mask', good_looking_mask[unpad_slice], optim_iter)
            self.summary.add_image('Train SDF', sdf_train.data[unpad_slice], optim_iter)
            self.summary.add_image('Train Contour', cont_train.data[unpad_slice], optim_iter)
            self.summary.add_image('Predicted Mask', torch.sigmoid(img_mask_out.data[unpad_slice]), optim_iter)
            self.summary.add_image('Predicted SDF', torch.sigmoid(img_sdf_out.data[unpad_slice]), optim_iter)
            self.summary.add_image('Predicted Contour', torch.sigmoid(img_cont_out.data[unpad_slice]), optim_iter)
            self.summary.add_image('Predicted Score', torch.sigmoid(scores), optim_iter)
            self.summary.add_image('Predicted Mask Patches', make_grid(pred_masks.data[:16].sigmoid(), nrow=4), optim_iter)
            self.summary.add_image('Target Mask Patches', make_grid(target_masks.data[:16].sigmoid(), nrow=4), optim_iter)
            self.summary.add_image('Target Image Patches', make_grid(img_crops[:16].sigmoid(), nrow=4), optim_iter)
            self.summary.add_image('True Mask Patches', make_grid(true_masks[:16].sigmoid(), nrow=4), optim_iter)

        self.add_scalar_averaged('Batch Time', batch_time, optim_iter)
        self.add_scalar_averaged('Score FScore', score_fscore, optim_iter)
        self.add_scalar_averaged('Score Precision', score_prec, optim_iter)
        self.add_scalar_averaged('Score Recall', score_rec, optim_iter)
        self.add_scalar_averaged('Mask IoU Full', f_iou.mean(), optim_iter)
        self.add_scalar_averaged('Mask IoU Threshold', t_iou.mean(), optim_iter)
        self.add_scalar_averaged('Box IoU', box_iou.mean(), optim_iter)
        self.add_scalar_averaged('Learning Rate', self.optimizer_gen.param_groups[0]['lr'], optim_iter)
        for b_i, b_c in enumerate(batch_masks):
            self.add_scalar_averaged(f'Masks At Level {b_i}', b_c, optim_iter)

        self.optim_iter += 1

        return loss

    def add_scalar_averaged(self, tag, scalar_value, global_step):
        if isinstance(scalar_value, Variable):
            scalar_value = scalar_value.data.view(1)[0]
        hist_size = 50
        if tag not in self.scalar_history:
            self.scalar_history[tag] = []
        hist = self.scalar_history[tag]
        hist.append(scalar_value)
        if len(hist) == hist_size:
            self.summary.add_scalar(tag, np.mean(hist), global_step)
            hist.clear()


def get_train_pairs(
        model, labels, sdf, img, net_out,
        pos_sdf_threshold=0.2, neg_sdf_threshold=-0.2,
        pos_size_limit=0.5, neg_size_limit=1,
        pos_samples=32, neg_to_pos_ratio=3):
    outputs = []
    for sample_idx in range(labels.shape[0]):
        net_out_sample = [(m[sample_idx], s[sample_idx], b[sample_idx]) for m, s, b in net_out]
        o = get_train_pairs_single(
            model, labels[sample_idx, 0], sdf[sample_idx, 0], img[sample_idx], net_out_sample,
            model.mask_pixel_sizes, pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limit, neg_size_limit,
            pos_samples, neg_to_pos_ratio
        )
        outputs.extend(o)

    if len(outputs) == 0:
        return None

    outputs = list(zip(*outputs))
    outputs, layer_idx = outputs[:-1], np.concatenate(outputs[-1])
    outputs = [torch.cat(o, 0) for o in outputs]
    return (*outputs, layer_idx)


def get_train_pairs_single(model, labels, sdf, img, net_out, pixel_sizes,
                           pos_sdf_threshold, neg_sdf_threshold,
                           pos_size_limit, neg_size_limit,
                           pos_samples, neg_to_pos_ratio):
    box_mask = get_object_boxes(labels, 2)
    resampled_layers = resample_data(labels, sdf, box_mask, img, pixel_sizes)
    outputs = []
    for layer_idx, layer_data in enumerate(zip(net_out, resampled_layers)):
        (out_masks, out_scores, out_boxes), (res_labels, res_sdf, res_boxes, res_img) = layer_data

        o = generate_samples_for_layer(
            model, out_masks, out_scores, out_boxes, res_labels, res_sdf, res_boxes, res_img,
            pos_sdf_threshold, neg_sdf_threshold,
            pos_size_limit, neg_size_limit,
            pos_samples, neg_to_pos_ratio
        )
        if o is not None:
            outputs.append((*o, o[0].shape[0] * [layer_idx]))
    return outputs


def generate_samples_for_layer(model, out_features, out_scores, out_boxes, labels, sdf, true_boxes, img,
                               pos_sdf_threshold, neg_sdf_threshold,
                               pos_size_limit, neg_size_limit,
                               pos_samples, neg_to_pos_ratio):
    assert out_boxes.dim() == 3 and out_boxes.shape[0] == 6
    assert true_boxes.dim() == 3 and true_boxes.shape[0] == 10
    assert out_boxes.shape[1] == out_boxes.shape[2]

    # border and stride for converting between image space and conv-center space
    stride = model.mask_size // model.region_size
    border = stride // 2

    # slice to select values from image at conv center locations
    mask_centers_slice = (
        slice(border, -border + 1, stride),
        slice(border, -border + 1, stride))
    # [fs, fs] - sdf at conv centers
    sdf_fs = sdf[mask_centers_slice]
    # [fs, fs] - labels at conv centers
    labels_fs = labels[mask_centers_slice].contiguous()
    # [(y, x, cov00, cov01, cov10, cov11), fs, fs] - obj boxes at conv centers
    true_boxes_fs = true_boxes[(slice(None), *mask_centers_slice)].contiguous()

    assert true_boxes_fs.shape[-1] == out_features.shape[-1], \
        (true_boxes_fs.shape, out_boxes.shape, out_features.shape, labels.shape, sdf.shape, img.shape)

    anchor_box_size = model.region_size / out_boxes.shape[-1]
    max_pos_size, min_pos_size = anchor_box_size * 2 ** pos_size_limit, anchor_box_size * 2 ** -pos_size_limit
    max_neg_size, min_neg_size = anchor_box_size * 2 ** neg_size_limit, anchor_box_size * 2 ** -neg_size_limit

    true_box_area = true_boxes_fs[2] * true_boxes_fs[3]
    true_box_size = true_box_area ** 0.5

    # assert true_box_size.min() >= 0

    pos_centers_fs = (true_box_size > min_pos_size) & \
                     (true_box_size < max_pos_size) & \
                     (sdf_fs > pos_sdf_threshold) & \
                     (labels_fs != 0)
    neg_centers_fs = (true_box_size > max_neg_size) | \
                     (true_box_size < min_neg_size) | \
                     (sdf_fs < neg_sdf_threshold)

    # TODO: allow zero negative centers
    if pos_centers_fs.sum() == 0 or neg_centers_fs.sum() == 0:
        return None

    pos_centers_fs_idx_all = pos_centers_fs.view(-1).nonzero().squeeze()
    neg_centers_fs_idx_all = neg_centers_fs.view(-1).nonzero().squeeze()
    pos_centers_fs_perm = torch.randperm(len(pos_centers_fs_idx_all))
    # neg_centers_fmap_perm = torch.randperm(len(neg_centers_fmap_idx_all))
    pos_centers_fs_perm = pos_centers_fs_perm[:pos_samples].contiguous().cuda()
    # neg_centers_fmap_perm = neg_centers_fmap_perm[:len(pos_centers_fmap_perm) * neg_to_pos_ratio].contiguous().cuda()
    pos_centers_fs_idx = pos_centers_fs_idx_all[pos_centers_fs_perm]
    # neg_centers_fmap_idx = neg_centers_fmap_idx_all[neg_centers_fmap_perm]

    pred_pos_scores = out_scores.take(Variable(pos_centers_fs_idx_all))
    pred_neg_scores = out_scores.take(Variable(neg_centers_fs_idx_all))
    pred_scores = torch.cat([pred_pos_scores, pred_neg_scores])
    target_scores = out_scores.data.new(pred_scores.shape[0]).fill_(0)
    target_scores[:pred_pos_scores.shape[0]] = 1

    # (NPos, [y, x, h, w, sin, cos])
    pred_boxes = out_boxes.view(out_boxes.shape[0], -1)[:, pos_centers_fs_idx_all].t()
    # (NPos, 2, 2)
    pred_cov = torch.bmm(create_scale_mat(pred_boxes[:, 2:4]), create_rot_mat(pred_boxes[:, 4:6]))
    pred_cov = torch.bmm(pred_cov.transpose(1, 2), pred_cov)
    pred_pos = pred_boxes[:, :2]
    # (NPos, [y, x, cov00, cov01, cov10, cov11])
    pred_boxes = torch.cat([pred_pos, pred_cov.view(-1, 4)], 1)
    # (NPos, [y, x, cov00, cov01, cov10, cov11])
    target_boxes = true_boxes_fs.view(true_boxes_fs.shape[0], -1)[:, pos_centers_fs_idx_all].t()
    target_boxes = torch.cat([target_boxes[:, :2], target_boxes[:, 6:]], 1)

    # (NPos, [y, x, h, w, sin, cos])
    mask_boxes = out_boxes.data.view(out_boxes.shape[0], -1)[:, pos_centers_fs_idx].t()

    pred_features = roi_align(out_features.unsqueeze(0), mask_boxes, model.region_size)
    img_crops = roi_align(
        Variable(img.unsqueeze(0), volatile=True),
        mask_boxes,
        model.mask_size
    ).data

    # NPos - labels at center of true boxes
    pos_center_label_nums = labels_fs.take(pos_centers_fs_idx)
    # assert (pos_center_label_nums == 0).sum() == 0
    target_masks = labels_to_mask_roi_align(labels, mask_boxes, pos_center_label_nums, model.mask_size, False)

    true_mask_boxes = true_boxes_fs.view(true_boxes_fs.shape[0], -1)[:, pos_centers_fs_idx].t()
    true_mask_boxes = true_mask_boxes[:, :6]
    true_masks = labels_to_mask_roi_align(labels, true_mask_boxes, pos_center_label_nums, model.mask_size, True)

    return pred_features, target_masks, pred_scores, target_scores, pred_boxes, target_boxes, img_crops, true_masks


def create_scale_mat(yx):
    assert yx.dim() == 2 and yx.shape[1] == 2
    mat = Variable(yx.data.new(yx.shape[0], 2, 2).zero_())
    mat[:, 0, 0] = yx[:, 1]
    mat[:, 1, 1] = yx[:, 0]
    return mat


def create_rot_mat(sincos):
    assert sincos.dim() == 2 and sincos.shape[1] == 2
    mat = Variable(sincos.data.new(sincos.shape[0], 2, 2).zero_())
    mat[:, 0, 1] = sincos[:, 0]
    mat[:, 1, 0] = -sincos[:, 0]
    mat[:, 0, 0] = mat[:, 1, 1] = sincos[:, 1]
    return mat


def get_object_boxes(labels, downsampling=1):
    # [src_size, src_size]
    assert labels.dim() == 2
    assert labels.shape[0] == labels.shape[1]

    # if downsampling != 1:
    #     labels = downscale_nonzero(labels, downsampling)

    channels = 10
    size = labels.shape[0]
    count = labels.max()
    if count == 0:
        return torch.zeros(channels, size, size).cuda()#torch.zeros(6, size * downsampling, size * downsampling).cuda()

    boxes = torch.zeros(channels, size, size).cuda()

    for label_num in range(1, count + 1):
        mask = labels == label_num
        # [num_px, 2]
        px_pos = mask.nonzero().float().add_(0.5).div_(size)
        if len(px_pos) < 9 or ((px_pos != px_pos[0]).sum(0) != 0).sum() < 2:
            continue
        # [2]
        mean_pos = px_pos.mean(0)
        diff = px_pos - mean_pos
        cov = diff.t() @ diff
        cov *= ((1 + box_padding) ** 2) * (4 ** 2) / (px_pos.shape[0] - 1)

        scale, rot = cov2x2_to_scale_rot(cov)

        # [(pos_y, pos_x, scale_max, scale_min, sin, cos)]
        # all relative to 0-1 image space
        vec = torch.Tensor([*mean_pos, *scale, math.sin(rot), math.cos(rot), *cov.view(4)]).cuda()
        boxes += vec.view(channels, 1, 1).expand_as(boxes) * mask.unsqueeze(0).float()

    # if downsampling != 1:
    #     boxes = F.upsample(boxes.unsqueeze(0), scale_factor=downsampling, mode='nearest').data.squeeze(0)

    return boxes


def cov2x2_to_scale_rot(cov):
    assert cov.shape == (2, 2)
    cov = cov.cpu().numpy()
    e, v = np.linalg.eig(cov)
    orig_e, orig_v = e, v
    if e[0] < e[1]:
        e = e[::-1]
        angle90 = np.pi / 2
        v = v @ np.array([[math.cos(angle90), math.sin(angle90)], [-math.sin(angle90), math.cos(angle90)]])
    assert np.all(e > -1e-4), (cov, e, orig_e, v, orig_v)
    scale = e.clip(min=1e-8) ** 0.5
    m = (v * scale).T
    angle = -math.atan2(-m[0, 1], m[0, 0])
    if angle < -math.pi / 2:
        angle += math.pi
    if angle > math.pi / 2:
        angle -= math.pi
    return scale.tolist(), angle


def resample_data(labels, sdf, box_mask, img, pixel_sizes):
    # labels - [size, size]
    # sdf - [size, size]
    # box_mask - [8, size, size]
    # img - [3, size, size]
    # pixel_sizes - [layers count]
    assert labels.shape == sdf.shape
    assert labels.dim() == 2
    resampled = []
    for px_size in pixel_sizes:
        assert labels.shape[-1] % px_size == 0
        assert px_size >= 1
        if px_size == 1:
            res_labels, res_sdf, res_boxes, res_img = labels, sdf, box_mask, img
        # elif px_size < 1:
        #     assert round(1 / px_size, 3) == int(1 / px_size)
        #     factor = int(1 / px_size)
        #     res_labels = F.upsample(labels.view(1, 1, *labels.shape).float(), scale_factor=factor).data[0, 0].long()
        #     res_boxes = F.upsample(box_mask.unsqueeze(0), scale_factor=factor).data[0] / px_size
        #     res_sdf = F.upsample(sdf.view(1, 1, *sdf.shape), scale_factor=factor, mode='bilinear').data[0, 0]
        #     res_img = F.upsample(img.unsqueeze(0), scale_factor=factor, mode='bilinear').data[0]
        else:
            res_labels = downscale_nonzero(labels, px_size)
            res_boxes = downscale_nonzero(box_mask, px_size)
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
    assert gx.shape[-1] == 1
    gx = gx.squeeze(-1)
    gx = gx.view(*x.shape[:-2], ds, ds)
    assert gx.shape == (*x.shape[:-2], ds, ds), (x.shape, gx.shape)
    return gx


def labels_to_mask_roi_align(labels, boxes, label_nums, crop_size, check_mask):
    assert labels.dim() == 2
    assert boxes.dim() == 2
    assert label_nums.dim() == 1
    masks = []
    for box, label_num in zip(boxes, label_nums):
        full_mask = (labels == label_num).view(1, 1, *labels.shape).float()
        # assert not check_mask or full_mask.sum() > 9
        mask = roi_align(Variable(full_mask, volatile=True), box.unsqueeze(0), crop_size).data
        # assert not check_mask or mask.sum() > 9, \
        #     (box, (labels == label_num).nonzero().float().add_(0.5).div_(labels.shape[-1]).mean(0), labels.shape)
        masks.append(mask)
    return torch.cat(masks, 0)


def rotated_box_iou(first_boxes, second_boxes):
    def mean_cov_to_mean_scale_rot(x):
        x = x.cpu()
        vecs = []
        for point in x:
            mean, cov = point[:2], point[2:].contiguous().view(2, 2)
            scale, rot = cov2x2_to_scale_rot(cov)
            vec = [*mean, *scale, math.sin(rot), math.cos(rot)]
            vecs.append(vec)
        return np.array(vecs)
    first_boxes, second_boxes = mean_cov_to_mean_scale_rot(first_boxes), mean_cov_to_mean_scale_rot(second_boxes)
    ious = []
    for a, b in zip(first_boxes, second_boxes):
        a, b = [(*v[:4], math.atan2(v[4], v[5]) / math.pi * 180) for v in (a, b)]
        inter = intersection_area(a, b)
        union = a[2] * a[3] + b[2] * b[3] - inter
        ious.append(inter / union)
    return torch.Tensor(ious)

