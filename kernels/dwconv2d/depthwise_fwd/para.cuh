#pragma once
namespace {
#define DIVUP(x, y) (((x) + (y)-1) / (y))

//
struct Param {
    int batch, src_chl, src_h, src_w, flt_h, flt_w, out_h, out_w, pad_h, pad_w;};

template <int x_, int y_>
struct ThreadConfig {
    static int const thread_x = x_;
    static_assert((thread_x & (thread_x - 1)) == 0, "thread_x must be pow of 2!");
    static int const thread_y = y_;
    static int const nr_threads = x_ * y_;
};

template <typename ThreadConfig_, int unroll_w_>
struct OutTileConfig {
    using ThreadConfig = ThreadConfig_;
    static int const unroll_h = 1;
    static int const unroll_w = unroll_w_;
    static int const block_h = ThreadConfig::thread_y * unroll_h;
    static int const block_w = ThreadConfig::thread_x * unroll_w;
    static int const row_size = unroll_h * block_w;
};

template <int fh_, int fw_>
struct FilterTileConfig {
    static int const unroll_h = 1;
    static int const unroll_w = DIVUP(fw_, 4);
    static int const block_w = fw_>9?fw_:9;  //为了编译器优化 权宜之计 宽若小于9一律按9处理
    static int const block_h = fh_;
    static int const unroll_size = unroll_h * unroll_w;
};

template <typename OutTileConfig_, typename FilterTileConfig_>
struct SrcTileConfig {
    using OutTileConfig = OutTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    static int const unroll_h = (OutTileConfig::unroll_h - 1) + FilterTileConfig::unroll_h;
    static int const unroll_w = (OutTileConfig::unroll_w - 1) + FilterTileConfig::block_w;
    static int const block_w = (OutTileConfig::block_w - 1) + FilterTileConfig::block_w;
    static int const block_h = (OutTileConfig::block_h - 1) + FilterTileConfig::block_h;
    static int const unroll_size = unroll_h * unroll_w;
    static int const unroll_w_out = OutTileConfig::unroll_w;
};

    
template <typename SrcTileConfig_>
struct SrcTileCount {
    using SrcTileConfig = SrcTileConfig_;
    static int const smem_h = SrcTileConfig::block_h;///
    static int const smem_w = SrcTileConfig::block_w;
    static int const smem_w_bank = smem_w+((SrcTileConfig::unroll_w_out==8)? 1:0);
    static int const smem_size = smem_h * smem_w_bank;
};

template <typename FilterTileConfig_>
struct FilterTileCount {
    using FilterTileConfig = FilterTileConfig_;
    static int const smem_w = FilterTileConfig::block_w;//DIVUP(FilterTileConfig::block_w, 2) * 2;
    static int const smem_h = FilterTileConfig::block_h;//
    static int const smem_w_bank = smem_w + 1;
    static int const smem_size = smem_h * smem_w_bank;
};

}  // namespace