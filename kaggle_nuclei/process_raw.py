from pathlib import Path

import numpy as np
import torch
from skimage import io
from tqdm import tqdm

from .distance_field import distance_field


def process_raw(file_path, has_mask=True, save_human_readable_masks=False):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
        imgs = []
        for image in (file / 'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs) == 1, file
        if img.shape[2] > 3:
            assert (img[:, :, 3] != 255).sum() == 0
        img = img[:, :, :3]

        if save_human_readable_masks:
            io.imsave(file_path / '..' / f'{"test_" if not has_mask else ""}{file.name}_img.png', img)

        if has_mask:
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
            if save_human_readable_masks:
                img_mask = mask.clone()
                img_mask[img_mask != 0] += 10
                io.imsave(file_path / '..' / f'{file.name}_mask.png', (img_mask * (1 / (masks.shape[0] + 10))).clip(0, 1))
            item['mask'] = torch.from_numpy(mask)
            item['distance_field'] = torch.from_numpy(distance_field(item['mask']).astype(np.float32))
        item['name'] = str(file).split('/')[-1]
        item['img'] = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        datas.append(item)
    return datas
