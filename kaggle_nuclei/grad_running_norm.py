import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GradRunningNorm(nn.Module):
    def __init__(self, weight=1, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.weight = weight
        self.register_buffer('avg_sq', None)
        self.avg_sq = torch.zeros(1)
        self.step = 0
        self.var_clone = VariableClone()
        self.var_clone.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        assert len(grad_input) == 1, grad_input
        grad_input, = grad_input
        self.avg_sq += (1 - self.momentum) * (grad_input.data.pow(2).mean() - self.avg_sq)
        self.step += 1
        bias_correction = 1 - self.momentum ** self.step
        std = Variable(self.avg_sq.div(bias_correction).add_(1e-8).sqrt_())
        return self.weight * grad_input / std,

    def forward(self, input):
        return self.var_clone(input)


class VariableClone(nn.Module):
    def forward(self, input):
        return input + 0
