import math
from functools import reduce

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from ..dataset import resnet_norm_mean, resnet_norm_std, train_pad
from .training import center_crop

mean_std_sub = torch.FloatTensor([resnet_norm_mean, resnet_norm_std]).cuda()
img_size_div = 32
border_pad = 64
total_pad = train_pad + border_pad


def predict(model, raw_data, stride=4, max_stride=32, max_scale=1, tested_scales=1):
    model.eval()

    max_scale = math.log10(max_scale)
    scales = np.logspace(-max_scale, max_scale, num=tested_scales)

    results = []
    for data in tqdm(raw_data):
        img = data['img']
        out_mean = None
        sum_count = 0
        for scale in scales:
            out = extract_proposals_from_image(model, img, scale, 0)
            return out
            if out[0] is None:
                continue
            out = reduce(np.add, out)
            out /= 4
            if out_mean is not None:
                out_mean += out
            else:
                out_mean = out
            sum_count += 1
        out_mean /= sum_count
        results.append(out_mean)
    return results


def masked_non_max_suppression(img, proposals, mask_threshold=0, max_allowed_intersection=0.2):
    labels = torch.LongTensor(*img.shape[1:]).zero_()
    proposals = sorted(proposals, key=lambda x: -x[1])
    cur_obj_index = 1
    for mask, _, pos in proposals:
        bounds_label = np.array([pos[0], pos[0] + mask.shape[0], pos[1], pos[1] + mask.shape[1]])
        bounds_label_clip = bounds_label.copy()
        bounds_label_clip[:2] = bounds_label[:2].clip(0, labels.shape[0])
        bounds_label_clip[2:] = bounds_label[2:].clip(0, labels.shape[1])
        bounds_mask_clip = bounds_label_clip - bounds_label
        bounds_label_clip, bounds_mask_clip = bounds_label_clip.tolist(), bounds_mask_clip.tolist()

        label_crop = labels[bounds_label_clip[0]:bounds_label_clip[1], bounds_label_clip[2]:bounds_label_clip[3]]
        mask_crop = mask[bounds_mask_clip[0]:bounds_mask_clip[1], bounds_mask_clip[2]:bounds_mask_clip[3]]
        label_crop_mask, mask_crop_mask = label_crop > 0, mask_crop > mask_threshold

        intersection_area = (label_crop_mask & mask_crop_mask).sum()
        mask_area = mask_crop_mask.sum()
        if intersection_area > mask_area * max_allowed_intersection:
            continue
        label_crop[(label_crop_mask == 0) & (mask_crop_mask != 0)] = cur_obj_index
        cur_obj_index += 1
    return labels


def extract_strided_proposals_from_image(model, img, scale=1, score_threshold=0.8,
                                         stride_start=-15, stride_end=16, stride=5):
    proposals = []
    for offset_y in range(stride_start, stride_end, stride):
        for offset_x in range(stride_start, stride_end, stride):
            new = extract_proposals_from_image(model, img, scale, score_threshold, (offset_y, offset_x))
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
    x = np.pad(x, padding, mode='reflect')
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0).cuda()
    x = (x - mean_std_sub[0].view(1, -1, 1, 1)) / mean_std_sub[1].view(1, -1, 1, 1)
    # x = flips[flip_type](x)
    # noise = x.new(1, 1, *x.shape[2:]).normal_(0, 1)
    # x = torch.cat([x, noise], 1)

    out_layers, out_img = model(Variable(x, volatile=True), train_pad)
    real_scale = s_shape / np.array(img.shape[:2])
    preds = [extract_proposals_from_layer(model, ly, psz, str, real_scale, score_threshold)
             for ly, psz, str in zip(out_layers, model.mask_pixel_sizes, model.mask_strides)]
    preds = [p for p in preds if p is not None]
    preds = [(m, s, (p - np.array(pad_offset) - border_pad) / real_scale)
             for masks, scores, positions in preds
             for m, s, p in zip(masks, scores, positions)]
    return preds


def extract_proposals_from_layer(model, layer, pixel_size, stride, scale, sigmoid_score_threshold):
    masks, scores = layer[0][0], layer[1].data[0, 0]
    logit_score_threshold = logit(sigmoid_score_threshold)
    good_score_mask = scores > logit_score_threshold
    good_idx = good_score_mask.view(-1).nonzero().squeeze()
    if len(good_idx) == 0:
        return None
    good_masks = center_crop(masks, good_idx, (0, model.conv_size), scores.shape[-1])
    good_masks = model.predict_masks(good_masks)
    size = (np.array(good_masks.shape[-2:]) * pixel_size / scale).round().astype(int)
    size = size[0].item(), size[1].item()
    good_masks = F.upsample(good_masks, size, mode='bilinear').data.squeeze(1)
    good_scores = scores.view(-1)[good_idx]
    good_positions = good_score_mask.nonzero() * stride #+ round(FPN.mask_size // 2 * pixel_size)
    # good_positions = mask_to_indexes(good_score_mask, stride, FPN.mask_size // 2 * pixel_size, image_size)
    # good_positions = torch.stack([good_idx / scores.shape[-1], good_idx % scores.shape[-2]], 1)
    # good_positions = good_positions * stride
    return good_masks.cpu().numpy(), good_scores.cpu().numpy(), good_positions.cpu().numpy()


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