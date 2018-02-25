import numpy as np
from skimage.transform import resize
from torch.autograd import Variable
import torch.nn.functional as F
from torchsample.utils import th_affine2d
import torch
from skimage import io
import scipy.misc
import math
from tqdm import tqdm


def predict(model, raw_data, max_scale=4, tested_scales=15, pad=32):
    model.eval()

    max_scale = math.log10(max_scale)
    scales = np.logspace(-max_scale, max_scale, num=tested_scales)

    results = []
    for data in tqdm(raw_data):
        img = data['img'].numpy().transpose(1, 2, 0)
        out_mean = np.zeros((*img.shape[:2], 3))
        sum_count = 0
        for scale in scales:
            s_shape = (np.array(img.shape) * scale / 16).round().astype(int) * 16 + pad * 2
            if np.max(s_shape) > 2048:
                # print(f'ignoring scale {scale}, size {tuple(s_shape)}, '
                #       f'source size {tuple(img.shape)} for {data["name"][:16]}')
                continue
            x = scipy.misc.imresize(img, s_shape).astype(np.float32) / 255
            x = x - x.mean()
            x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x).unsqueeze(0).cuda()
            out = []
            flips = [
                lambda v: v,
                lambda v: flip(v, 2),
                lambda v: flip(v, 3),
                lambda v: flip(flip(v, 3), 2)
            ]
            for f in flips:
                s = f(x)
                s = F.pad(s, 4 * (pad,), mode='reflect')
                s = model(Variable(s.data.contiguous(), volatile=True)).data
                s = s[:, :, pad:-pad, pad:-pad]
                s = f(s).cpu()
                out.append(s)
            out = torch.cat(out, 0).sum(0).div_(4)
            out = out.cpu().numpy()
            out = np.stack([scipy.misc.imresize(o, img.shape[:2], mode='F') for o in out], 2)
            out_mean += out
            sum_count += 1
        out_mean /= sum_count
        results.append(out_mean)
    return results


# https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    x = x.contiguous()
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def predict_upsample(model, dataloader, raw_data):
    preds_test_upsampled = []
    model.eval()
    sample_idx = 0
    for data in dataloader:
        data = Variable(data.cuda(), volatile=True)
        out = model(data).data.cpu().numpy()
        for o in out:
            size = raw_data[sample_idx]['img'].shape[:2]
            preds_test_upsampled.append(
                resize(np.squeeze(o), size, mode='constant', preserve_range=True))
            sample_idx += 1
    return preds_test_upsampled