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


def extract_proposals_chunked(model, img, score_threshold):
    img = img.numpy().transpose(1, 2, 0)
    s_shape = (np.array(img.shape[:2]) / img_size_div).round().astype(int) * img_size_div

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


def extract_proposals_from_chunk(model, img, score_threshold):
    x = torch.from_numpy(img).cuda()
    x = tsf.Normalize(mean=resnet_norm_mean, std=resnet_norm_std)(x)
    x = x.unsqueeze(0)
    # x = flips[flip_type](x)
    # noise = x.new(1, 1, *x.shape[2:]).normal_(0, 1)
    # x = torch.cat([x, noise], 1)

    out_layers, out_img = model(Variable(x, volatile=True), train_pad)
    preds = [extract_proposals_from_chunk_layer(model, ly, score_threshold) for ly in out_layers]
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