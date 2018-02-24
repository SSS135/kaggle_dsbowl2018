import random
import torch.nn.functional as F


class MeanNormalize:
    def __call__(self, input):
        return input.sub(input.mean())


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        h_idx = random.randint(0, input.shape[-2] - self.size[0])
        w_idx = random.randint(0, input.shape[-1] - self.size[1])
        return input[:, h_idx: (h_idx + self.size[0]), w_idx: (w_idx + self.size[1])]


class Pad:
    def __init__(self, size, mode='constant'):
        self.size = size
        self.mode = mode

    def __call__(self, x):
        return F.pad(x, pad=self.size, mode=self.mode).data