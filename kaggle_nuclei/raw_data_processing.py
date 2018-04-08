from pathlib import Path

import numpy as np
import scipy.misc
import skfmm
import torch
from scipy import ndimage
from skimage import io
from tqdm import tqdm
import torch.nn.functional as F


def process_raw(root, has_mask=True, save_human_readable_masks=False):
    root = Path(root)
    files = sorted(list(Path(root).iterdir()))
    datas = []
    human_readable_root = root / '../merged_masks'

    for file in tqdm(files):
        try:
            item = {'name': str(file).split('/')[-1]}
            img = None
            for image in (file / 'images').iterdir():
                assert img is None, file
                img = io.imread(image)
            if img.shape[2] > 3:
                assert (img[:, :, 3] != 255).sum() == 0
            img = img[:, :, :3]

            if save_human_readable_masks:
                io.imsave(human_readable_root / f'{"test_" if not has_mask else ""}{file.name}_img.png', img)

            if has_mask:
                mask, num_objects = generate_mask(file)
                sdf = calc_distance_field(mask)
                median_mask_area = calc_median_mask_area(mask)
                # obj_boxes = generate_object_boxes(mask)

                if save_human_readable_masks:
                    img_mask = mask.copy()
                    img_mask[img_mask != 0] += 10
                    io.imsave(human_readable_root / f'{file.name}_mask.png', (img_mask * (1 / (num_objects + 10))).clip(0, 1))
                    io.imsave(human_readable_root / f'{file.name}_sdf.png', (sdf * 0.5 + 0.5).clip(0, 1))

                sdf_compressed = (sdf * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                mask_compressed = mask if num_objects >= 255 else mask.astype(np.uint8)
                # H x W
                item['mask_compressed'] = torch.from_numpy(mask_compressed)
                # H x W
                item['sdf_compressed'] = torch.from_numpy(sdf_compressed)
                # 4 x H x W
                # item['obj_boxes'] = torch.from_numpy(obj_boxes.transpose((2, 0, 1)))
                item['median_mask_area'] = median_mask_area
            # 3 x H x W
            item['img'] = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            datas.append(item)
        except:
            print(file)
            raise
    return datas


# def generate_object_boxes(labels):
#     objs = [x for x in ndimage.find_objects(labels) if x is not None]
#     obj_infos = [(
#         y.start,
#         x.start,
#         y.stop - y.start,
#         x.stop - x.start,
#     ) for y, x in objs]
#     obj_infos = np.array(obj_infos)
#     info_mask = np.zeros((*labels.shape, 4), dtype=np.int32)
#     for label_idx, rect in enumerate(obj_infos, 1):
#         info_mask[labels == label_idx] = rect
#     return info_mask


def generate_mask(file):
    mask_files = list((file / 'masks').iterdir())
    masks = None
    for label_num, mask in enumerate(mask_files, 1):
        mask = io.imread(mask)
        assert (mask[(mask != 0)] == 255).all(), file
        if masks is None:
            H, W = mask.shape
            masks = np.zeros((H, W), np.int32)
        masks[mask != 0] = label_num
    # tmp_mask = masks.sum(0)
    # assert (tmp_mask[tmp_mask != 0] == 255).all(), file
    # for label_num, mask in enumerate(masks):
    #     masks[label_num] = mask // 255 * (label_num + 1)
    # mask = masks.sum(0)
    return masks, len(mask_files)


def calc_distance_field(labels, clip=(-1, 1), downscale=2):
    h, w = src_shape = labels.shape
    masks_count = labels.max()

    if h % downscale != 0 or w % downscale != 0:
        h_pad, w_pad = downscale - h % downscale, downscale - w % downscale
        labels = np.pad(labels, ((0, h_pad), (0, w_pad)), 'edge')
        h, w = labels.shape
        assert h % downscale == 0 and w % downscale == 0
    padded_shape = labels.shape

    maxdist = np.full(np.array(labels.shape) // downscale, -np.inf, dtype=np.float32)
    dists = []
    for layer in range(masks_count):
        mask = labels == (layer + 1)
        mask = mask.reshape(h // downscale, downscale, w // downscale, downscale).mean(axis=(1, 3))
        if mask.max() < 0.5:
            continue
        phi = mask * 2 - 1
        # phi = np.ones(mask.shape)
        # phi[mask > 0] = 1
        # phi[mask <= 0] = -1
        dist = skfmm.distance(phi)
        dists.append(dist)
    assert len(dists) != 0

    mean_max_dist = np.max(dists, axis=(1, 2)).mean()
    for dist in dists:
        dist[dist > 0] /= dist.max()
        dist[dist < 0] /= mean_max_dist
        np.maximum(maxdist, dist, out=maxdist)

    maxdist = maxdist.clip(*clip)
    maxdist = scipy.misc.imresize(maxdist, padded_shape, mode='F')
    maxdist = maxdist[:src_shape[0], :src_shape[1]]
    assert maxdist.shape == src_shape
    return maxdist


def calc_median_mask_area(labels):
    labels = torch.from_numpy(labels)
    areas = []
    for label_num in range(1, labels.max() + 1):
        mask = labels == label_num
        # ignore very small objects
        if mask.sum() < 9:
            continue
        # [num_px, 2]
        px_pos = mask.nonzero()
        # check for all equal x or y positions and skip them.
        # without this check sometimes determinant == 0.
        if ((px_pos != px_pos[0]).sum(0) != 0).sum() < 2:
            continue
        px_pos = px_pos.float().add_(0.5)
        assert len(px_pos) != 0
        # [2]
        mean_pos = px_pos.mean(0)
        cov = (px_pos - mean_pos).t() @ (px_pos - mean_pos) / (px_pos.shape[0] - 1)
        cov *= 16
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        assert det > 0
        area = det ** 0.5
        areas.append(area)
    assert len(areas) != 0
    return np.median(areas)