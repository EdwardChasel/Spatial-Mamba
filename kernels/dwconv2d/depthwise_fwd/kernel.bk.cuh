#pragma once
#include "para.cuh"
#include <iostream>

namespace {

template <typename ThreadConfig_, typename TileCount_>
struct Global2SharedMem {
    using TileCount = TileCount_;
    using ThreadConfig = ThreadConfig_;
    const int tid = threadIdx.y * ThreadConfig::thread_x + threadIdx.x;

    float* smem;
    const float* g_ptr;
    int stride;
    int start_h, start_w, bound_h, bound_w;
    

    __device__ __forceinline__ Global2SharedMem(
            float* smem_, int stride_, int s_h, int s_w, int b_h, int b_w);
    __device__ __forceinline__ void copy();
    __device__ __forceinline__ float* sh_ptr_as_copy_t(int y, int x) {
        return smem + y * TileCount::smem_w_bank + x;
    }
};

template <typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__
Global2SharedMem<ThreadConfig_, TileCount_>::Global2SharedMem(
        float* smem_, int stride_, int s_h, int s_w, int b_h, int b_w)
        : smem(smem_),
          stride(stride_),
          start_h(s_h),
          start_w(s_w),
          bound_h(b_h),
          bound_w(b_w) {}

template <typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<ThreadConfig_, TileCount_>::copy() {
    static int const load_w = TileCount::smem_w > 32 ? 32 : TileCount::smem_w;
    static int const load_h = ThreadConfig::nr_threads / load_w;
    static int const num_free_thread = ThreadConfig::nr_threads - load_w*load_h;
    static int const h_per_thread = DIVUP(TileCount::smem_h, load_h);
    static int const w_per_thread = DIVUP(TileCount::smem_w, load_w);
    static bool constexpr check_bounds_h = (TileCount::smem_h % load_h != 0)||(num_free_thread != 0);
    static bool constexpr check_bounds_w = TileCount::smem_w % load_w != 0;
    const int y_base_idx = tid / load_w;
    const int x_base_idx = tid - y_base_idx * load_w;
#pragma unroll
    for (int i = 0; i < h_per_thread; ++i) {
        int smem_h_idx = y_base_idx + i * load_h;
        int src_h_idx;
        src_h_idx = start_h + smem_h_idx;
        if (check_bounds_h && smem_h_idx >= TileCount::smem_h)
            break;
#pragma unroll
        for (int j = 0; j < w_per_thread; ++j) {
            int smem_w_idx = x_base_idx + j * load_w;
            int src_w_idx;
            src_w_idx = start_w + smem_w_idx;
            if (check_bounds_w && smem_w_idx >= TileCount::smem_w)
                break;
            float val = 0.0f;
            if (src_h_idx >= 0 && src_h_idx < bound_h && src_w_idx >= 0 &&
                src_w_idx < bound_w) {
                val = g_ptr[src_h_idx * stride + src_w_idx];
            }
            *(sh_ptr_as_copy_t(smem_h_idx, smem_w_idx)) = val;
        }
    }
}

// CUDA kernel to compute the depthwise convolution forward pass in NCHW format,
// tailored for small images up to 32x32. Stride and depth multiplier must be 1.
// Padding must be 'SAME', which allows to reuse the index computation. Only
// use this kernel if CanLaunchDepthwiseConv2dGPU(args) returns true.
// Tiles of the input and filter tensors are loaded into shared memory before
// performing the convolution. Each thread handles two elements per iteration,
// one each in the lower and upper half of a tile.
// Backprop input direction is the same as forward direction with the filter
// rotated by 180°.

template <typename ThreadConfig_, typename FilterTileConfig_, typename OutTileConfig_, typename SrcTileConfig_, typename SrcTileCount_, typename FilterTileCount_>
__global__ void dwconv2dkernelfp32(
        const Param param, const float* input, const float* filter, const float* bias, float* output, const bool is_bias) {

    using ThreadConfig = ThreadConfig_;
    using SrcTileConfig = SrcTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    using OutTileConfig = OutTileConfig_;
    using SrcTileCount = SrcTileCount_;
    using FilterTileCount = FilterTileCount_;

    using SrcGlobal2ShareVisitor = Global2SharedMem<ThreadConfig, SrcTileCount>;
    using FilterGlobal2ShareVisitor = Global2SharedMem<ThreadConfig, FilterTileCount>;

    int off_batch_channel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(float) <= 8, "Insufficient alignment detected");
    float* smem_src = reinterpret_cast<float*>(smem);
    float* smem_flt = reinterpret_cast<float*>(&smem_src[SrcTileCount::smem_size]);

    int off_channel = off_batch_channel % param.src_chl,
        out_start_h = off_obh * OutTileConfig::block_h,
        out_start_w = off_obw * OutTileConfig::block_w,
        src_start_h = out_start_h - param.pad_h,
        src_start_w = out_start_w - param.pad_w,
        out_base_h_idx = out_start_h + off_oh * OutTileConfig::unroll_h;

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src, // src data
            static_cast<int>(param.src_w),
            static_cast<int>(src_start_h),
            static_cast<int>(src_start_w),
            static_cast<int>(param.src_h),
            static_cast<int>(param.src_w)};

    FilterGlobal2ShareVisitor gl2sh_flt = {
            smem_flt,
            static_cast<int>(param.flt_w),
            0,
            0,
            static_cast<int>(param.flt_h),
            static_cast<int>(param.flt_w)};

    gl2sh_src.g_ptr = input + off_batch_channel * param.src_h * param.src_w;
    gl2sh_flt.g_ptr = filter + off_channel * param.flt_h * param.flt_w;

    gl2sh_src.copy();
    gl2sh_flt.copy();

    float* bias_share;

    if (is_bias)
    {
        bias_share = reinterpret_cast<float*>(&smem_src[SrcTileCount::smem_size + FilterTileCount::smem_size]);
        if (off_oh==0&&off_ow==0){
        bias_share[0] = bias[off_channel];}
    }
    __syncthreads();
    if (out_base_h_idx >= param.out_h) {
        return ;
    }

    float* smem_src_ptr = smem_src + off_ow * OutTileConfig::unroll_w;
    float* smem_flt_ptr = smem_flt;// + off_ow * FilterTileConfig::unroll_w;
    float* out_base_ptr = output + off_batch_channel * param.out_h * param.out_w;

    float reg_src[SrcTileConfig::unroll_h * SrcTileConfig::unroll_w];
    float reg_flt[FilterTileConfig::unroll_h * FilterTileConfig::block_w];
    float sum[OutTileConfig::row_size] = {0.0};

    for (int fh = 0; fh < FilterTileConfig::block_h; fh += 1) {//为了编译器优化 权宜之计 高不小于7
        #pragma unroll
        for (int sw = 0; sw < SrcTileConfig::unroll_w; ++sw) {
            reg_src[sw] = smem_src_ptr[(off_oh + fh) * SrcTileCount::smem_w_bank + sw];
        }

        #pragma unroll
        for (int fw = 0; fw < FilterTileConfig::block_w; ++fw) {
            reg_flt[fw] = smem_flt_ptr[(fh) * FilterTileCount::smem_w_bank + fw];
        }

        #pragma unroll
        for (int fw = 0; fw < FilterTileConfig::block_w; ++fw) {
            #pragma unroll
            for (int ow = 0; ow < OutTileConfig::unroll_w; ++ow) {
                sum[ow] += reg_flt[fw] * reg_src[fw + ow];
            }
        }
    }

    int start_roll_w = out_start_w + off_ow*OutTileConfig::unroll_w;

    float thread_bias;
    if(is_bias)
    {
        thread_bias = bias_share[0];
    }
    
    #pragma unroll
    for (int j = 0; j < OutTileConfig::unroll_w; ++j) {
        int out_w_idx = start_roll_w + j;
        if (out_w_idx >= param.out_w)
            return;
        if (is_bias){
            out_base_ptr[out_base_h_idx * param.out_w + out_w_idx] = sum[j] + thread_bias;
                }
        else{
            out_base_ptr[out_base_h_idx * param.out_w + out_w_idx] = sum[j];}
    }
}

template <int fh, int fw, int unroll_ow>
void Dwconv2dKernelFP32(
        const Param& param, const float* input, const float* filter, const float* bias, float* output, const bool is_bias) {

    
    using ThreadConfig = ThreadConfig<4, 32>;
    using FilterTileConfig = FilterTileConfig<fh, fw>;
    using OutTileConfig = OutTileConfig<ThreadConfig, unroll_ow>;
    using SrcTileConfig = SrcTileConfig<OutTileConfig, FilterTileConfig>;
    using SrcTileCount = SrcTileCount<SrcTileConfig>;
    using FilterTileCount = FilterTileCount<FilterTileConfig>;

    dim3 block(ThreadConfig::thread_x, ThreadConfig::thread_y);
    dim3 grid;

    // std::cout<<SrcTileCount::smem_w<<std::endl;
    // std::cout<<SrcTileCount::smem_w_bank<<std::endl;
    // std::cout<<OutTileConfig::unroll_w<<std::endl;
    
    grid.x = param.batch * param.src_chl;
    grid.y = DIVUP(param.out_w, OutTileConfig::block_w);
    grid.z = DIVUP(param.out_h, OutTileConfig::block_h);
    const int shared_storage = (SrcTileCount::smem_size + FilterTileCount::smem_size + 1) * sizeof(float);

    void (*kernel)(const Param, const float*, const float*, const float*, float*, const bool);
    kernel = dwconv2dkernelfp32<ThreadConfig, FilterTileConfig, OutTileConfig, SrcTileConfig, SrcTileCount, FilterTileCount>;
    kernel<<<grid, block, shared_storage>>>(param, input, filter, bias, output, is_bias);
}


#define INSTANCE_ABC(a, b, c)\
    if (param.out_w <= c * 4||c==8) {\
        Dwconv2dKernelFP32<a, b, c>(param, src_, flt_, bias_, dst_, is_bias);\
    }


#define INSTANCE_AB(a, b)\
if (param.flt_w > b-2) {\
    INSTANCE_ABC(a, b, 4)\
    else INSTANCE_ABC(a, b, 8)\
}

#define INSTANCE_A(a)\
if (param.flt_h > a-2) {\
    INSTANCE_AB(a, 31) \
    else INSTANCE_AB(a, 29) \
    else INSTANCE_AB(a, 27) \
    else INSTANCE_AB(a, 25) \
    else INSTANCE_AB(a, 23) \
    else INSTANCE_AB(a, 21) \
    else INSTANCE_AB(a, 19) \
    else INSTANCE_AB(a, 17) \
    else INSTANCE_AB(a, 15) \
    else INSTANCE_AB(a, 13) \
    else INSTANCE_AB(a, 11) \
    else INSTANCE_AB(a, 9) \
    else INSTANCE_AB(a, 7) \
    else INSTANCE_AB(a, 5) \
    else INSTANCE_AB(a, 3) \
}


// #define INSTANCE_A(a)\
// if (param.flt_w > a-2) {\
//     INSTANCE_ABC(a, a, 4)\
//     else INSTANCE_ABC(a, a, 8)\
// }

#define INSTANCE \
    INSTANCE_A(31) \
    else INSTANCE_A(29) \
    else INSTANCE_A(27) \
    else INSTANCE_A(25) \
    else INSTANCE_A(23) \
    else INSTANCE_A(21) \
    else INSTANCE_A(19) \
    else INSTANCE_A(17) \
    else INSTANCE_A(15) \
    else INSTANCE_A(13) \
    else INSTANCE_A(11) \
    else INSTANCE_A(9) \
    else INSTANCE_A(7) \
    else INSTANCE_A(5) \
    else INSTANCE_A(3)



}  // anonymous namespace
