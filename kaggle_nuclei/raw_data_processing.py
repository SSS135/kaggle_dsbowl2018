from pathlib import Path

import numpy as np
import torch
from skimage import io
from tqdm import tqdm
from scipy import ndimage

from .distance_field import distance_field


def process_raw(root, has_mask=True, save_human_readable_masks=False):
    root = Path(root)
    files = sorted(list(Path(root).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
        img = None
        for image in (file / 'images').iterdir():
            assert img is None, file
            img = io.imread(image)
        if img.shape[2] > 3:
            assert (img[:, :, 3] != 255).sum() == 0
        img = img[:, :, :3]

        if save_human_readable_masks:
            io.imsave(root / '..' / f'{"test_" if not has_mask else ""}{file.name}_img.png', img)

        if has_mask:
            mask, num_objects = generate_mask(file)
            sdf = distance_field(mask)
            info_mask = generate_object_infos(mask)

            if save_human_readable_masks:
                img_mask = mask.clone()
                img_mask[img_mask != 0] += 10
                io.imsave(root / '..' / f'{file.name}_mask.png', (img_mask * (1 / (num_objects + 10))).clip(0, 1))

            sdf_compressed = (sdf * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            mask_compressed = mask if num_objects >= 255 else mask.astype(np.uint8)
            item['mask_compressed'] = torch.from_numpy(mask_compressed)
            item['sdf_compressed'] = torch.from_numpy(sdf_compressed)
            item['info_mask'] = torch.from_numpy(info_mask.transpose((2, 0, 1)))
        item['name'] = str(file).split('/')[-1]
        item['img'] = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        datas.append(item)
    return datas


def generate_object_infos(labels):
    objs = [x for x in ndimage.find_objects(labels) if x is not None]
    obj_infos = [(
        (y.start + y.stop) / 2,
        (x.start + x.stop) / 2,
        max(y.stop - y.start, x.stop - x.start)
    ) for y, x in objs]
    obj_infos = np.array(obj_infos)
    info_mask = np.zeros((*labels.shape, 3), dtype=np.int32)
    for label_idx, rect in enumerate(obj_infos, 1):
        info_mask[labels == label_idx] = rect
    return info_mask


def generate_mask(file):
    mask_files = list((file / 'masks').iterdir())
    masks = None
    for ii, mask in enumerate(mask_files):
        mask = io.imread(mask)
        assert (mask[(mask != 0)] == 255).all()
        if masks is None:
            H, W = mask.shape
            masks = np.zeros((len(mask_files), H, W), np.int32)
        masks[ii] = mask
    tmp_mask = masks.sum(0)
    assert (tmp_mask[tmp_mask != 0] == 255).all()
    for ii, mask in enumerate(masks):
        masks[ii] = mask // 255 * (ii + 1)
    mask = masks.sum(0)
    return mask, len(mask_files)
