import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from Dwconv.dwconv_layer import *
 
cudnn.benchmark = True

# this is the test code for the speed of the new conv2d

size = 64  #input size
batch = 64 #batch size
 
def benchmark_my_conv(ksize, batch=batch, dim=384, img_size=size, depth=24):
    m = nn.Sequential(
        *[DwConv2d(dim, kernel_size=(ksize, ksize), padding=(ksize//2, ksize//2), bias=False) for _ in range(depth)]
    ).cuda()
    x = torch.rand(batch, dim, img_size, img_size).cuda()

    for i in range(5):
        t = time.perf_counter()
        y = m(x)
        y.sum().backward()
        torch.cuda.synchronize()
        t = time.perf_counter() - t
    return t * 1000

def benchmark_pytorch(ksize, batch=batch, dim=384, img_size=size, depth=24):
    m = nn.Sequential(
        *[nn.Conv2d(dim, dim, kernel_size=(ksize, ksize), padding=(ksize//2, ksize//2), groups=dim, bias=False) for _ in range(depth)]
    ).cuda()
    x = torch.rand(batch, dim, img_size, img_size).cuda()

    for i in range(5):
        t = time.perf_counter()
        y = m(x)
        y.sum().backward()
        torch.cuda.synchronize()
        t = time.perf_counter() - t
    return t * 1000
 
if __name__ == "__main__":
    args = dict()
    kernel_size_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    
    
    print('**************new dwconv2d****************')
    for k in kernel_size_list:
        t = benchmark_my_conv(k, **args)
        print("kernel size", k, "iter time", t, "ms")
    print('*******************************************')

    print('**************pytorch dwconv2d****************')
    for k in kernel_size_list:
        t = benchmark_pytorch(k, **args)
        print("kernel size", k, "iter time", t, "ms")
    print('*******************************************')
