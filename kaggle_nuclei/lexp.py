import torch
import torch.autograd


class LExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.exp()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def lexp(x):
    return LExpFunction.apply(x)