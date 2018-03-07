import math

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from functools import reduce
from skimage.transform import resize
from torch.autograd import Variable
from tqdm import tqdm
from .dataset import resnet_norm_mean, resnet_norm_std, train_pad
from .feature_pyramid_network import FPN


mean_std_sub = torch.FloatTensor([resnet_norm_mean, resnet_norm_std]).cuda()
img_size_div = 32
border_pad = 64
total_pad = train_pad + border_pad


def predict_rpn(model, raw_data, stride=4, max_stride=32, max_scale=1, tested_scales=1):
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

    out_layers = model(Variable(x, volatile=True), train_pad)
    real_scale = s_shape / np.array(img.shape[:2])
    preds = [extract_proposals_from_layer(ly, psz, str, real_scale, score_threshold)
             for ly, psz, str in zip(out_layers, model.mask_pixel_sizes, model.mask_strides)]
    preds = [p for p in preds if p is not None]
    preds = [(m, s, (p - np.array(pad_offset) - border_pad) / real_scale)
             for masks, scores, positions in preds
             for m, s, p in zip(masks, scores, positions)]
    return preds


def extract_proposals_from_layer(layer, pixel_size, stride, scale, sigmoid_score_threshold):
    masks, scores = [x.data for x in layer]
    logit_score_threshold = logit(sigmoid_score_threshold)
    good_idx = (scores > logit_score_threshold).view(-1).nonzero().squeeze()
    if len(good_idx) == 0:
        return None
    good_masks = masks.view(-1, *masks.shape[-2:]).index_select(0, good_idx)
    good_masks = Variable(good_masks.unsqueeze(1), volatile=True)
    size = (np.array(masks.shape[-2:]) * pixel_size / scale).round().astype(int)
    size = size[0].item(), size[1].item()
    good_masks = F.upsample(good_masks, size, mode='bilinear').data.squeeze(1)
    good_scores = scores.view(-1)[good_idx]
    good_positions = torch.stack([good_idx / masks.shape[-3], good_idx % masks.shape[-3]], 1)
    good_positions = good_positions * stride
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