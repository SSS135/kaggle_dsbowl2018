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


mean_std_sub = torch.FloatTensor([resnet_norm_mean, resnet_norm_std]).cuda()


def predict(model, raw_data, max_scale=2, tested_scales=7):
    model.eval()

    max_scale = math.log10(max_scale)
    scales = np.logspace(-max_scale, max_scale, num=tested_scales)

    results = []
    for data in tqdm(raw_data):
        img = data['img']
        out_mean = None
        sum_count = 0
        for scale in scales:
            out = [predict_single(model, img, scale, flip_type=f) for f in range(4)]
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


def predict_single(model, img, scale=1, flip_type: 0 or 1 or 2 or 3=0):
    div = 32
    img = img.numpy().transpose(1, 2, 0)
    s_shape = (np.array(img.shape) * scale / div).round().astype(int) * div + train_pad * 2
    if np.max(s_shape) > 2048:
        # print(f'ignoring scale {scale}, size {tuple(s_shape)}, '
        #       f'source size {tuple(img.shape)} for {data["name"][:16]}')
        return None

    x = scipy.misc.imresize(img, s_shape).astype(np.float32) / 255
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0).cuda()
    x = (x - mean_std_sub[0].view(1, -1, 1, 1)) / mean_std_sub[1].view(1, -1, 1, 1)
    x = flips[flip_type](x)
    x = F.pad(x, 4 * (train_pad,), mode='reflect').data
    # noise = x.new(1, 1, *x.shape[2:]).normal_(0, 1)
    # x = torch.cat([x, noise], 1)
    x = model(Variable(x, volatile=True)).data
    x = flips[flip_type](x).cpu()
    x = x[0, :, train_pad:-train_pad, train_pad:-train_pad]
    x = x.cpu().numpy()
    x = np.stack([scipy.misc.imresize(o, img.shape[:2], mode='F') for o in x], 2)
    return x


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