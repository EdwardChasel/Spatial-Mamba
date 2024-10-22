from torch import nn
import torch
import dwconv2d


class DepthwiseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, padding_h, padding_w, is_bias):
        w_inv = torch.flip(w, dims=[2, 3])
        ctx.save_for_backward(x, w_inv)
        ctx.padding_h = padding_h
        ctx.padding_w = padding_w
        ctx.is_bias = is_bias
        if is_bias:
            return dwconv2d.dwconv2dbias_fp32(x, w, b, padding_h, padding_w)
        else:
            return dwconv2d.dwconv2d_fp32(x, w, padding_h, padding_w)

    @staticmethod
    def backward(ctx, grad):
        padding_h, padding_w, is_bias = ctx.padding_h, ctx.padding_w, ctx.is_bias
        x, w_inv = ctx.saved_tensors
        if not grad.is_contiguous():
            grad = grad.contiguous()
        dx = dwconv2d.dwconv2d_fp32_dgrad(grad, w_inv, padding_h, padding_w)
        dw = dwconv2d.dwconv2d_fp32_wgrad(x, grad, padding_h, padding_w)
        if is_bias:
            db = dwconv2d.dwconv2d_fp32_bgrad(grad)
            return dx, dw, db, None, None, None
        else:
            return dx, dw, None, None, None, None

class DwConv2d(nn.Module):
    def __init__(self, num_channel, kernel_size, padding, bias=True) -> None:
        super().__init__()


        kernel_size_h, kernel_size_w = kernel_size
        self.padding_h, self.padding_w = padding

        weight = torch.randn(num_channel, 1, kernel_size_h, kernel_size_w)
        self.weight = nn.Parameter(weight)
        self.is_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channel))
        else:
            self.bias = None
        
    def forward(self, x):
        y = DepthwiseFunction.apply(x, self.weight, self.bias, self.padding_h, self.padding_w, self.is_bias)
        return y
    




if __name__=='__main__':
    # layer = nn.Sequential(
    #     *[DwConv2d(384, 13, padding=13//2) for _ in range(24)]
    # ).cuda()
    layer = DwConv2d(384, (13,13), padding=(13//2,13//2)).cuda()
    input = torch.rand((8, 384, 64, 64)).cuda()
    y = layer(input)
    # y.sum().backward()
    print(y.shape)
    from copy import deepcopy
    model = deepcopy(layer)
    y = model(input)
    print(y.shape)
