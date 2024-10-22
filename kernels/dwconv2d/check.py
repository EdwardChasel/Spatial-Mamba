import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from Dwconv.dwconv_layer import *
 
cudnn.benchmark = True

# this is the test code for the speed of the new conv2d

size = 32  #input size
batch = 2 #batch size
dim = 16

if __name__ == "__main__":
    args = dict()

    strided_conv1 = torch.nn.Conv2d(dim, dim, kernel_size=3, dilation=1, padding=1, groups=dim, padding_mode="replicate", bias=False).cuda()
    strided_conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, dilation=3, padding=3, groups=dim, padding_mode="replicate", bias=False).cuda()
    strided_conv3 = torch.nn.Conv2d(dim, dim, kernel_size=3, dilation=5, padding=5, groups=dim, padding_mode="replicate", bias=False).cuda()

    inp = torch.rand(batch, dim, size, size).cuda()

    out1 = strided_conv1(inp)
    out2 = strided_conv2(inp)
    out3 = strided_conv3(inp)
    out = out1+out2+out3

    weight = torch.zeros(dim, 1, 11, 11).float().cuda()
    weight[:, :, 4:7, 4:7] = strided_conv1.weight

    weight[:, :, 2:3, 2:3] = strided_conv2.weight[:,:,0:1,0:1]
    weight[:, :, 2:3, 5:6] = strided_conv2.weight[:,:,0:1,1:2]
    weight[:, :, 2:3, 8:9] = strided_conv2.weight[:,:,0:1,2:3]
    weight[:, :, 5:6, 2:3] = strided_conv2.weight[:,:,1:2,0:1]
    weight[:, :, 5:6, 5:6] += strided_conv2.weight[:,:,1:2,1:2]
    weight[:, :, 5:6, 8:9] = strided_conv2.weight[:,:,1:2,2:3]
    weight[:, :, 8:9, 2:3] = strided_conv2.weight[:,:,2:3,0:1]
    weight[:, :, 8:9, 5:6] = strided_conv2.weight[:,:,2:3,1:2]
    weight[:, :, 8:9, 8:9] = strided_conv2.weight[:,:,2:3,2:3]

    weight[:, :, 0:1, 0:1] = strided_conv3.weight[:,:,0:1,0:1]
    weight[:, :, 0:1, 5:6] = strided_conv3.weight[:,:,0:1,1:2]
    weight[:, :, 0:1, 10:11] = strided_conv3.weight[:,:,0:1,2:3]
    weight[:, :, 5:6, 0:1] = strided_conv3.weight[:,:,1:2,0:1]
    weight[:, :, 5:6, 5:6] += strided_conv3.weight[:,:,1:2,1:2]
    weight[:, :, 5:6, 10:11] = strided_conv3.weight[:,:,1:2,2:3]
    weight[:, :, 10:11, 0:1] = strided_conv3.weight[:,:,2:3,0:1]
    weight[:, :, 10:11, 5:6] = strided_conv3.weight[:,:,2:3,1:2]
    weight[:, :, 10:11, 10:11] = strided_conv3.weight[:,:,2:3,2:3]

    _out = DepthwiseFunction.apply(inp, weight, None, 11//2, 11//2, False)

    print(f"diff - {torch.mean(torch.abs(out-_out))}")
