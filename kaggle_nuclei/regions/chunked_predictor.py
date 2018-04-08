import numpy as np
import scipy.misc
import torch
import torchvision.transforms as tsf
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import math

from ..roi_align import roi_align
from ..settings import resnet_norm_mean, resnet_norm_std, train_pad, box_padding, train_size
from ..utils import unpad


def extract_proposals_transformed(model, img, scale, rotation, hflip, vflip, score_threshold, max_chunk_proposals=0):
    transf_matrix = get_affine_matrix(scale, rotation, hflip, vflip)
    tr_img = affine_transform_image(img, transf_matrix)
    masks, scores, boxes = extract_proposals(model, tr_img, score_threshold, max_chunk_proposals)


def extract_proposals(model, img, score_threshold, max_chunk_proposals):
    assert img.dim() == 3 and img.shape[0] == 3

    nc, raw_h, raw_w = img.shape
    crops_count = np.ceil(np.array(img.shape[:2]) / train_size).astype(int)
    padded_size = crops_count * train_size + train_pad * 2
    padded_h, padded_w = padded_size.tolist()
    assert padded_h >= 0 and padded_w >= 0

    h_pad, w_pad = padded_h - raw_h, padded_w - raw_w
    pad_t, pad_l = h_pad // 2, w_pad // 2
    pad_b, pad_r = h_pad - pad_t, w_pad - pad_l
    assert all(p >= 0 for p in (pad_t, pad_b, pad_l, pad_r))
    img = median_pad2d(img, pad_t, pad_b, pad_l, pad_r)

    all_masks, all_scores, all_boxes = [], [], []
    for h_crop_idx, w_crop_idx in itertools.product(range(crops_count[0]), range(crops_count[1])):
        crop_side = train_size + train_pad * 2
        crop_t, crop_l = h_crop_idx * train_size, w_crop_idx * train_size
        crop_b, crop_r = crop_t + crop_side, crop_l + crop_side
        crop = img[:, crop_t:crop_b, crop_l:crop_l].contiguous()
        crop_masks, crop_scores, crop_boxes = extract_proposals_from_chunk(
            model, crop, score_threshold, max_chunk_proposals)
        crop_boxes[:, 0] += crop_t
        crop_boxes[:, 1] += crop_l
        all_masks.append(crop_masks)
        all_scores.append(crop_scores)
        all_boxes.append(crop_boxes)
    all_masks = torch.cat(all_masks, 0)
    all_scores = torch.cat(all_scores, 0)
    all_boxes = torch.cat(all_boxes, 0)

    return all_masks, all_scores, all_boxes


def extract_proposals_from_chunk(model, img, score_threshold, max_chunk_proposals):
    x = torch.from_numpy(img).cuda()
    x = tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std)(x)
    x = x.unsqueeze(0)
    # x = flips[flip_type](x)
    # noise = x.new(1, 1, *x.shape[2:]).normal_(0, 1)
    # x = torch.cat([x, noise], 1)

    out_layers, out_img = model(Variable(x, volatile=True))
    preds = [extract_proposals_from_chunk_layer(model, ly, score_threshold) for ly in out_layers]
    preds = [p for p in preds if p is not None]
    preds = [(m, s, p - (np.array(pad_offset) + border_pad) / real_scale)
             for masks, scores, boxes in preds
             for m, s, b in zip(masks, scores, boxes)]
    preds = [(m, s, b) for m, s, b in preds if
             p[0] + m.shape[0] > 1 and
             p[1] + m.shape[1] > 1 and
             p[0] < img.shape[0] - 2 and
             p[1] < img.shape[1] - 2]
    return preds


def extract_proposals_from_chunk_layer(model, layer, sigmoid_score_threshold, max_proposals=512):
    features, scores, boxes = layer[0].data, layer[1].data[0], layer[2][0].data[0]
    valid_idx = scores.view(-1).topk(max_proposals)[1]
    valid_idx = valid_idx[scores.view(-1)[valid_idx] > sigmoid_score_threshold]

    if len(valid_idx) == 0:
        return [], [], []

    assert boxes.shape[0] == 4
    valid_scores = scores.view(-1)[valid_idx]
    valid_boxes = boxes.contiguous().view(4, -1)[:, valid_idx].t()
    valid_boxes = pad_boxes(valid_boxes, box_padding)
    masks = batched_mask_prediction(model, features, valid_boxes)

    return masks, valid_scores.cpu().numpy(), valid_boxes.cpu().numpy()


def batched_mask_prediction(model, features, boxes, batch_size=32):
    masks = []
    for boxes in boxes.split(batch_size, 0):
        # (NBox x C x RH x RW)
        feature_crops = roi_align(Variable(features, volatile=True), boxes, model.region_size)
        # (NBox x 1 x MH x MW)
        batch_masks = model.predict_masks(feature_crops)
        masks.extend(batch_masks.squeeze(1).cpu().numpy())
    return masks


def median_pad2d(img, pad_t, pad_b, pad_l, pad_r):
    median = img.view(img.shape[0], -1).median(-1)[0].view(img.shape[0], 1, 1)
    padded = F.pad(img.unsqueeze(0), (pad_l, pad_r, pad_t, pad_b)).data.squeeze(0)

    c, h, w = padded.shape
    padded[:, :pad_t] = median
    padded[:, h - pad_b:] = median
    padded[:, :, :pad_l] = median
    padded[:, :, w - pad_r:] = median

    return padded


def get_affine_matrix(scale, rot, hflip, vflip):
    rot_transform = np.array([
        [math.cos(rot), math.sin(rot), 0],
        [-math.sin(rot), math.cos(rot), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    scale_transform = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    flip_transform = np.array([
        [-1 if hflip else 1, 0, 0],
        [0, -1 if vflip else 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    return flip_transform @ rot_transform @ scale_transform


def affine_transform_image(img, matrix):
    transform = img.new(transform[:2]).cuda()

    grid = F.affine_grid(transform, torch.Size((1, img.shape[0], crop_size, crop_size)))
    input = input.expand(boxes.shape[0], -1, -1, -1)
    x = F.grid_sample(input, grid)
    return x