/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "fmha.h"
#include "fmha_fprop_kernel_1xN.h"
#include "fmha_fprop_kernel_1xN_nl.h"

using Kernel_traits = FMHA_kernel_traits< 768, 64, 16, 1, 8, 0x08u>;

extern "C" __global__ void fmha_fprop_fp16_768_64_sm80_train_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::device_1xN<Kernel_traits, true>(params);
}

extern "C" __global__ void fmha_fprop_fp16_768_64_sm80_predict_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::device_1xN<Kernel_traits, false>(params);
}

template<int CHUNKS>
__global__ void fmha_fprop_fp16_768_64_sm80_train_nl_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::device_1xN_nl<CHUNKS,Kernel_traits, true>(params);
}

template<int CHUNKS>
__global__ void fmha_fprop_fp16_768_64_sm80_predict_nl_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::device_1xN_nl<CHUNKS, Kernel_traits, false>(params);
}


void run_fmha_fp16_768_64_sm80(const Fused_multihead_attention_fprop_params &params, bool is_training, cudaStream_t stream) {

    auto kernel = is_training ? &fmha_fprop_fp16_768_64_sm80_train_kernel : &fmha_fprop_fp16_768_64_sm80_predict_kernel;

    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_o = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    constexpr int smem_size = smem_size_q + std::max(smem_size_v, smem_size_o + smem_size_softmax);
    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 grid(params.h, params.b);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, stream>>>(params);
}

void run_fmha_fp16_768_64_sm80_nl(const Fused_multihead_attention_fprop_params &params, const bool is_training, const int num_chunks, cudaStream_t stream) {

    auto kernel = is_training ? &fmha_fprop_fp16_768_64_sm80_train_nl_kernel<2> : &fmha_fprop_fp16_768_64_sm80_predict_nl_kernel<2>;
    if( num_chunks == 2 ) {
        kernel = is_training ? &fmha_fprop_fp16_768_64_sm80_train_nl_kernel<2>
                             : &fmha_fprop_fp16_768_64_sm80_predict_nl_kernel<2>;
    } else if( num_chunks == 3 ) {
        kernel = is_training ? &fmha_fprop_fp16_768_64_sm80_train_nl_kernel<3>
                             : &fmha_fprop_fp16_768_64_sm80_predict_nl_kernel<3>;
    } else if( num_chunks == 4 ) {
        kernel = is_training ? &fmha_fprop_fp16_768_64_sm80_train_nl_kernel<4>
                             : &fmha_fprop_fp16_768_64_sm80_predict_nl_kernel<4>;
    } else {
        assert(false && "Unsupported num_chunks");
    }

    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_o = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    constexpr int smem_size = smem_size_q + std::max(smem_size_v, smem_size_o + smem_size_softmax);
    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    dim3 grid(params.h, params.b, num_chunks);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, stream>>>(params);
}
