import torch
import torch.nn.functional as F
from torch.autograd import Variable


def roi_align(input, boxes, crop_size):
    """

    Args:
        input: feature map (1 x C x H x W)
        boxes: crop boxes in 0 - 1 range (N x [y, x, h, w])

    Returns: crops (N, C, `crop_size`, `crop_size`)

    """
    assert input.dim() == 4 and input.shape[0] == 1
    assert boxes.dim() == 2 and boxes.shape[1] == 4
    assert isinstance(input, Variable) and not isinstance(boxes, Variable)

    # boxes = boxes / torch.cuda.FloatTensor([input.shape[2], input.shape[3], input.shape[2], input.shape[3]])

    tr_y = boxes[:, 2].mul(0.5).add_(boxes[:, 0]).sub_(0.5).mul_(2)
    tr_x = boxes[:, 3].mul(0.5).add_(boxes[:, 1]).sub_(0.5).mul_(2)
    sc_y = boxes[:, 2]
    sc_x = boxes[:, 3]

    sc_mat = torch.eye(3).unsqueeze(0).cuda().repeat(boxes.shape[0], 1, 1)
    sc_mat[:, 0, 0] = sc_x
    sc_mat[:, 1, 1] = sc_y

    tr_mat = torch.eye(3).unsqueeze(0).cuda().repeat(boxes.shape[0], 1, 1)
    tr_mat[:, 0, 2] = tr_x
    tr_mat[:, 1, 2] = tr_y

    aff_mat = tr_mat @ sc_mat
    aff_mat = aff_mat[:, :2]

    grid = F.affine_grid(Variable(aff_mat), torch.Size((boxes.shape[0], input.shape[1], crop_size, crop_size)))
    input = input.expand(boxes.shape[0], -1, -1, -1)
    x = F.grid_sample(input, grid)
    return x


def pad_boxes(boxes, padding):
    assert boxes.dim() == 2 and boxes.shape[1] == 4
    boxes = boxes.t().clone()
    boxes[:2] -= padding * boxes[2:]
    boxes[2:] *= 1 + 2 * padding
    return boxes.t()