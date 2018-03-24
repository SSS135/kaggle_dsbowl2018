import math
from functools import reduce

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.autograd import Variable
from tqdm import tqdm
from ..settings import resnet_norm_mean, resnet_norm_std, train_pad, box_padding
import itertools
import torchvision.transforms as tsf
from ..roi_align import roi_align, pad_boxes

img_size_div = 32
border_pad = 64
total_pad = train_pad + border_pad


def masked_non_max_suppression(img_shape, proposals, mask_threshold=0, max_allowed_intersection=0.2, min_obj_size=10):
    labels = np.zeros(img_shape, dtype=int)
    used_proposals = []
    proposals = sorted(proposals, key=lambda x: -x[1])
    cur_obj_index = 1
    for mask, score, pos in proposals:
        pos = pos.round().astype(int)
        bounds_label = np.array([pos[0], pos[0] + mask.shape[0], pos[1], pos[1] + mask.shape[1]])
        bounds_label_clip = bounds_label.copy()
        bounds_label_clip[:2] = bounds_label[:2].clip(0, labels.shape[0])
        bounds_label_clip[2:] = bounds_label[2:].clip(0, labels.shape[1])
        bounds_mask_clip = bounds_label_clip - bounds_label + np.array([0, mask.shape[0], 0, mask.shape[1]])
        bounds_label_clip, bounds_mask_clip = bounds_label_clip.tolist(), bounds_mask_clip.tolist()

        label_crop = labels[bounds_label_clip[0]:bounds_label_clip[1], bounds_label_clip[2]:bounds_label_clip[3]]
        mask_crop = mask[bounds_mask_clip[0]:bounds_mask_clip[1], bounds_mask_clip[2]:bounds_mask_clip[3]]
        label_crop_mask, mask_crop_mask = label_crop > 0, mask_crop > mask_threshold

        # print(labels.shape, mask.shape, bounds_label, pos)
        # print(bounds_mask_clip, bounds_label_clip)

        intersection_area = (label_crop_mask & mask_crop_mask).sum()
        mask_area = mask_crop_mask.sum()
        if mask_area == 0 or intersection_area > mask_area * max_allowed_intersection:
            continue

        obj = ndimage.find_objects(mask > 0)
        if len(obj) != 1:
            continue
        obj = obj[0]
        shape = obj[0].stop - obj[0].start, obj[1].stop - obj[1].start
        if shape[0] < min_obj_size or shape[1] < min_obj_size:
            continue

        used_proposals.append((mask, score, pos))
        label_crop[(label_crop_mask == 0) & (mask_crop_mask != 0)] = cur_obj_index
        cur_obj_index += 1
    return labels, used_proposals


def extract_strided_proposals_from_image(model, img, score_threshold=0.8,
                                         max_scale=1, tested_scales=1,
                                         stride_start=-15, stride_end=16, stride=5):
    max_scale = math.log10(max_scale)
    scales = np.logspace(-max_scale, max_scale, num=tested_scales)
    proposals = []
    combs = itertools.product(
        range(stride_start, stride_end, stride),
        range(stride_start, stride_end, stride),
        scales)
    for (offset_y, offset_x, scale) in combs:
        new = extract_proposals_from_image(model, img, scale, score_threshold, (offset_y, offset_x))
        if new is not None:
            proposals.extend(new)
    return proposals


def extract_proposals_from_image(model, img, scale=1, score_threshold=0.8, pad_offset=(0, 0)):
    img = img.numpy().transpose(1, 2, 0)
    s_shape = (np.array(img.shape[:2]) * scale / img_size_div).round().astype(int) * img_size_div
    if np.max(s_shape) > 1024:
        # print(f'ignoring scale {scale}, size {tuple(s_shape)}, '
        #       f'source size {tuple(img.shape)} for {data["name"][:16]}')
        return None

    x = scipy.misc.imresize(img, s_shape).astype(np.float32) / 255
    padding = (total_pad + pad_offset[0], total_pad - pad_offset[0]), \
              (total_pad + pad_offset[1], total_pad - pad_offset[1]), \
              (0, 0)
    x = np.pad(x, padding, mode='median')
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).cuda()
    x = tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std)(x)
    x = x.unsqueeze(0)
    # x = flips[flip_type](x)
    # noise = x.new(1, 1, *x.shape[2:]).normal_(0, 1)
    # x = torch.cat([x, noise], 1)

    out_layers, out_img = model(Variable(x, volatile=True), train_pad)
    real_scale = s_shape / np.array(img.shape[:2])
    preds = [extract_proposals_from_layer(model, ly, out_img.shape[-2:], real_scale, score_threshold)
             for ly, psz, str in zip(out_layers, model.mask_pixel_sizes, model.mask_strides)]
    preds = [p for p in preds if p is not None]
    preds = [(m, s, p - (np.array(pad_offset) + border_pad) / real_scale)
             for masks, scores, positions in preds
             for m, s, p in zip(masks, scores, positions)]
    preds = [(m, s, p) for m, s, p in preds if
             p[0] + m.shape[0] > 1 and
             p[1] + m.shape[1] > 1 and
             p[0] < img.shape[0] - 2 and
             p[1] < img.shape[1] - 2]
    return preds


def extract_proposals_from_layer(model, layer, img_size, scale, sigmoid_score_threshold, max_proposals=512):
    features, scores, boxes = layer[0].data, layer[1].data[0], layer[2][0].data[0]
    valid_idx = scores.view(-1).topk(max_proposals)[1]
    valid_idx = valid_idx[scores.view(-1)[valid_idx] > sigmoid_score_threshold]
    # logit_score_threshold = logit(sigmoid_score_threshold)
    # valid_scores_mask = scores > logit_score_threshold
    # valid_idx = valid_scores_mask.view(-1).nonzero().squeeze()

    if len(valid_idx) == 0:
        return None

    valid_boxes = boxes.contiguous().view(4, -1)[:, valid_idx].t()
    valid_boxes = pad_boxes(valid_boxes, box_padding)
    resized_masks, valid_positions = batched_mask_prediction(model, features, valid_boxes, img_size, scale)

    valid_scores = scores.view(-1)[valid_idx]
    return resized_masks, valid_scores.cpu().numpy(), valid_positions


def batched_mask_prediction(model, features, boxes, img_size, scale, batch_size=32):
    resized_masks = []
    positions = []
    for boxes in boxes.split(batch_size, 0):
        # (NBox x C x RH x RW)
        feature_crops = roi_align(Variable(features, volatile=True), boxes, model.region_size)
        # (NBox x 1 x MH x MW)
        valid_masks = model.predict_masks(feature_crops)
        # (NBox x 4)
        boxes_px = boxes * boxes.new(2 * (np.asarray(img_size) / scale).tolist())
        boxes_px = boxes_px.round_().long()  # TODO: more accurate rounding

        for mask, box in zip(valid_masks, boxes_px):
            size = box[2:].cpu().numpy().tolist()
            pos = box[:2].cpu().numpy().tolist()
            # (H x W)
            mask = F.upsample(mask.unsqueeze(0), size, mode='bilinear').data.squeeze().cpu().numpy()
            resized_masks.append(mask)
            positions.append(pos)
    return resized_masks, np.array(positions)


def logit(x):
    return math.log(x / (1 - x))


# https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    x = x.contiguous()
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


flips = [
    lambda v: v,
    lambda v: flip(v, 2),
    lambda v: flip(v, 3),
    lambda v: flip(flip(v, 3), 2)
]