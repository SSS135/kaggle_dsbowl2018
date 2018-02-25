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
import torchvision.transforms as tsf


def predict(model, raw_data, max_scale=4, tested_scales=15, pad=32):
    model.eval()

    max_scale = math.log10(max_scale)
    scales = np.logspace(-max_scale, max_scale, num=tested_scales)

    mean_std_sub = torch.FloatTensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]).cuda()

    results = []
    for data in tqdm(raw_data):
        img = data['img'].numpy().transpose(1, 2, 0)
        out_mean = np.zeros((*img.shape[:2], 3))
        sum_count = 0
        for scale in scales:
            div = 32
            s_shape = (np.array(img.shape) * scale / div).round().astype(int) * div + pad * 2
            if np.max(s_shape) > 2048:
                # print(f'ignoring scale {scale}, size {tuple(s_shape)}, '
                #       f'source size {tuple(img.shape)} for {data["name"][:16]}')
                continue

            x = scipy.misc.imresize(img, s_shape).astype(np.float32) / 255
            # x = x - x.mean()
            x = x.transpose(2, 0, 1)
            x = torch.from_numpy(x).unsqueeze(0).cuda()
            x = (x - mean_std_sub[0].view(1, -1, 1, 1)) / mean_std_sub[1].view(1, -1, 1, 1)
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