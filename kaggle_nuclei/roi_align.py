import torch
import torch.nn.functional as F
from torch.autograd import Variable


def roi_align(input, boxes, crop_size):
    """

    Args:
        input: feature map (1 x C x H x W)
        boxes: (N x [y, x, h, w, sin, cos]) crop boxes in 0 - 1 space
        crop_size: output image size

    Returns: crops (N, C, `crop_size`, `crop_size`)

    """
    assert input.dim() == 4 and input.shape[0] == 1
    assert boxes.dim() == 2 and boxes.shape[1] == 6
    assert isinstance(input, Variable) and not isinstance(boxes, Variable)

    assert boxes[:, :2].max() < 2 and boxes[:, :2].min() > -1, boxes
    assert boxes[:, 2:4].max() < 2 and boxes[:, 2:4].min() > 1e-6, boxes
    assert boxes[:, 4:].max() <= 1 and boxes[:, 4:].min() >= -1, boxes

    eye = torch.eye(3).unsqueeze(0).cuda().repeat(boxes.shape[0], 1, 1)

    sc_mat = eye.clone()
    sc_mat[:, 0, 0] = boxes[:, 3]
    sc_mat[:, 1, 1] = boxes[:, 2]

    tr_mat = eye.clone()
    tr_mat[:, 0, 2] = boxes[:, 1] * 2 - 1
    tr_mat[:, 1, 2] = boxes[:, 0] * 2 - 1

    rot_mat = eye.clone()
    rot_mat[:, 0, 1] = boxes[:, 4]
    rot_mat[:, 1, 0] = -boxes[:, 4]
    rot_mat[:, 0, 0] = rot_mat[:, 1, 1] = boxes[:, 5]

    aff_mat = sc_mat @ rot_mat @ tr_mat
    aff_mat = aff_mat[:, :2]

    grid = F.affine_grid(Variable(aff_mat), torch.Size((boxes.shape[0], input.shape[1], crop_size, crop_size)))
    input = input.expand(boxes.shape[0], -1, -1, -1)
    x = F.grid_sample(input, grid)
    return x