#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <iostream>

#include "cuda.h"
#include "kernel.cuh"
#include "para.cuh"

using IntArrayRef = c10::ArrayRef<int64_t>;

at::Tensor Dwconv2dLauncherFP32(const at::Tensor input, const at::Tensor weight, const int padding_h, const int padding_w)
{
    const int N = input.size(0);
    const int C = input.size(1);
    const int iH = input.size(2);
    const int iW = input.size(3);

    const int kH = weight.size(2);
    const int kW = weight.size(3);

    const int oH = iH - kH + 1 + padding_h * 2;
    const int oW = iW - kW + 1 + padding_w * 2;

    const bool is_bias = 0;
    if (kH==3&&kW==3&&padding_h==1&&padding_w==1) 
    {
        // IntArrayRef kernel_size = {kH, kW};
        at::NoGradGuard no_grad;
        at::Tensor dst = at::cudnn_convolution(input, weight, {1, 1}, {1, 1}, {1, 1}, C, 1, 1, 1);
        return dst;
    }
    else
    {
        Param param = {N, C, iH, iW, kH, kW, oH, oW, padding_h, padding_w};
        at::Tensor dst = at::zeros({N, C, oH, oW}, input.options());
        const float *src_ = input.data_ptr<float>();
        const float *flt_ = weight.data_ptr<float>();
        const float *bias_ = NULL;
        float *dst_ = dst.data_ptr<float>();
        INSTANCE
        return dst;
    } 
}

at::Tensor Dwconv2dBiasLauncherFP32(const at::Tensor input, const at::Tensor weight, const at::Tensor bias, const int padding_h, const int padding_w)
{
    const int N = input.size(0);
    const int C = input.size(1);
    const int iH = input.size(2);
    const int iW = input.size(3);

    const int kH = weight.size(2);
    const int kW = weight.size(3);

    const int oH = iH - kH + 1 + padding_h * 2;
    const int oW = iW - kW + 1 + padding_w * 2;
    const bool is_bias = 1;

    if (kH==3&&kW==3&&padding_h==1&&padding_w==1&&is_bias==0) 
    {
        // IntArrayRef kernel_size = {kH, kW};
        at::NoGradGuard no_grad;
        at::Tensor dst = at::cudnn_convolution(input, weight, {1, 1}, {1, 1}, {1, 1}, C, 1, 1, 1);
        return dst;
    }
    else if(kH==3&&kW==3)
    {
        at::NoGradGuard no_grad;
        at::Tensor dst = at::conv2d(input, weight, bias, {1, 1}, {padding_h, padding_w}, {1, 1}, C);
        return dst;
    }
    else
    {
        Param param = {N, C, iH, iW, kH, kW, oH, oW, padding_h, padding_w};
        at::Tensor dst = at::zeros({N, C, oH, oW}, input.options());
        const float *src_ = input.data_ptr<float>();
        const float *flt_ = weight.data_ptr<float>();
        const float *bias_ = bias.data_ptr<float>();
        float *dst_ = dst.data_ptr<float>();
        
        INSTANCE
        return dst;
    } 
}

