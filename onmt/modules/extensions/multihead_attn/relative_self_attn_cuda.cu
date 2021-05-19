#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "THC/THC.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>

#include "strided_batched_gemm.h"
#include "softmax_dropout.h"
#include "dropout.h"

using namespace torch::indexing;

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

namespace multihead_attn {
namespace relative_self {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs,
                               torch::Tensor const& pos,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& input_biases,
                               torch::Tensor const& output_biases,
                               torch::Tensor const& r_w_bias,
                               torch::Tensor const& r_r_bias,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                  )
{
    const int   embed_dim         = inputs.size(2);
    const int   sequences         = inputs.size(1);
    const int   q_seq_len         = inputs.size(0);
    const int   k_seq_len         = q_seq_len;
    const int   batches           = sequences * q_seq_len;
    const int   head_dim          = embed_dim / heads;
    const int   output_lin_dim    = 3 * embed_dim;
    const int   attn_batches      = heads * sequences;
    const int   lead_dim_qkv      = attn_batches * 3 * head_dim;
    const int   lead_dim          = attn_batches * head_dim;
    const int   batch_stride_qkv  = 3 * head_dim;
    const int   batch_stride      = head_dim;
    const int   dropout_elems     = attn_batches * q_seq_len * k_seq_len;
    const float alpha             = 1.0;
    const float beta_zero         = 0.0;
    const float beta_one          = 1.0;
    const float scale             = 1.0 / sqrt(static_cast<float>(head_dim));

    // There is no reason to use more than one stream as every kernel is
    // sequentially dependent
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    // 3 Intermediate Results + Output (Note: dropout intermediates are generated by ATen library code)
    auto act_options  = inputs.options().requires_grad(false);
    auto mask_options = act_options.dtype(torch::kUInt8);

    torch::Tensor rr_head_q = torch::empty({q_seq_len, attn_batches, head_dim}, act_options);
    torch::Tensor rw_head_q = torch::empty({q_seq_len, attn_batches, head_dim}, act_options);
    torch::Tensor input_lin_results = torch::empty({q_seq_len, sequences, output_lin_dim}, act_options);
    torch::Tensor attn_scores       = torch::empty({attn_batches, q_seq_len, k_seq_len},   act_options);
    torch::Tensor attn_scores_bd     = torch::empty({q_seq_len, attn_batches, k_seq_len},   act_options);
    torch::Tensor softmax_results   = torch::empty({attn_batches, q_seq_len, k_seq_len},   act_options);
    torch::Tensor dropout_results   = torch::empty({attn_batches, q_seq_len, k_seq_len},   act_options);
    torch::Tensor dropout_mask      = torch::empty({attn_batches, q_seq_len, k_seq_len},   mask_options);
    torch::Tensor matmul2_results   = torch::empty({q_seq_len, attn_batches, head_dim},    act_options);
    torch::Tensor outputs           = torch::empty_like(inputs, act_options);

    // Input Linear Results Pointers to Q, K, and V of interviewed activations
    void* q_lin_results_ptr   = static_cast<void*>(input_lin_results.data_ptr());
    void* k_lin_results_ptr   = static_cast<void*>(static_cast<half*>(input_lin_results.data_ptr()) + head_dim);
    void* v_lin_results_ptr   = static_cast<void*>(static_cast<half*>(input_lin_results.data_ptr()) + 2*head_dim);

    torch::Tensor query = input_lin_results.view({q_seq_len, sequences*heads,
                                                  3, head_dim}).index({Slice(), Slice(), 0, Slice()});

    void* rw_head_q_ptr   = static_cast<void*>(rw_head_q.data_ptr());

    // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
    void* attn_scores_ptr = static_cast<void*>(attn_scores.data_ptr());
    void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());
    void* dropout_results_ptr = static_cast<void*>(dropout_results.data_ptr());

    char a_layout_t{'t'};
    char a_layout_n{'n'};
    char b_layout_n{'n'};
//    char b_layout_t{'t'};

    THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Input Linear Fwd
    input_lin_results.copy_(input_biases);
    THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             output_lin_dim,
                             batches,
                             embed_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(inputs.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(&beta_one),
                             q_lin_results_ptr,
                             CUDA_R_16F,
                             output_lin_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Just view and then copy (with broadcast)
    rr_head_q.view({q_seq_len * sequences, heads, head_dim}).copy_(r_r_bias.unsqueeze(0));
    rw_head_q.view({q_seq_len * sequences, heads, head_dim}).copy_(r_w_bias.unsqueeze(0));

    // both of those tensors have the same size with query now

    rw_head_q.add_(query);
    rr_head_q.add_(query);

    //    matmul_ac batched GEMMs
    //    rw_head_q: [len_q, bsz*heads, head_dim]
    //    keys:      [len_k, bsz*heads, head_dim]
    //    per_batch: [len_q x head_dim] \mul [head_dim, len_k]
    //    cublas-cm  [len_k, head_dim]  \mul [head_dim, len_q]
    //    result-cm  [len_k x len_q x bsz*heads]
    //    MatMul1 of Dot-Product Attention Plus scaling by 1/Sqrt(head size)

    // k_lin_results_ptr has to transpose because its [head_dim x [BH x len_k]] -> transpose to have head_dim last
    // rw_head_q [head_dim *  bsz*heads x len_q] -> no transpose
//    torch::Tensor matmul_ac = at::baddbmm(attn_scores, rw_head_q.transpose(0, 1),
//                               key.contiguous().view({k_seq_len, attn_batches, head_dim}).transpose(0, 1).transpose(1, 2),
//                               0.0, scale);

    const int   batch_count_ac   = attn_batches;
    gemm_switch_fp32accum(   state,
                             a_layout_t,  // a to be transposed
                             b_layout_n,  // b not transposed
                             k_seq_len,  // m = len_k
                             q_seq_len,  // n = len_q
                             head_dim,   // k = head_dim
                             scale,  // alpha
                             static_cast<const half*>(k_lin_results_ptr),
                             lead_dim_qkv, // attn_batches * head_dim * 3 because k is stored within qkv
                             batch_stride_qkv,  // 3 * head_dim
                             static_cast<const half*>(rw_head_q_ptr), // [ldb x n] = [attn_batches * head_dim x len_q]
                             lead_dim, // attn_batches * head_dim because rw_head_q is separated
                             batch_stride, // head_dim
                             beta_zero,
                             static_cast<half*>(attn_scores_ptr),
                             k_seq_len, // ldc
                             k_seq_len*q_seq_len, // c stride
                             batch_count_ac); // batch count

    //    matmul2 batched GEMMs
    //    queries+bias:  [len_q, bsz*heads, head_dim] B
    //    rel_positions: [len_q, len_k, head_dim] A
    //    per_batch: [bsz*heads x head_dim] \mul [head_dim, len_k]
    //    cublas-cm  [len_k x head_dim]  \mul [head_dim, bsz*heads]
    //    MatMul1 of Dot-Product Attention Plus scaling by 1/Sqrt(head size)

      // at:: alternative: inplace baddbmm
      attn_scores.transpose(0, 1).baddbmm_(rr_head_q, pos.transpose(1, 2), beta_one, scale);

//    attn_scores = attn_scores.transpose(0, 1).contiguous();
//    void* attn_scores_out_ptr = static_cast<void*>(attn_scores_out.data_ptr());

//    const int   batch_count_bd   = q_seq_len;
//    const int   lda = k_seq_len;  // k_seq_len or k_seq_len * q_seq_len
//    const int   batch_stride_a = k_seq_len * head_dim; // k_seq_len * head_dim or k_seq_len
//    const int   ldb = head_dim;  // head_dim or head_dim * len_q
//    const int   batch_stride_b = attn_batches * head_dim ; // attn_batches * head_dim or head_dim
//    const int   ldc = k_seq_len;  // k_seq_len * q_seq_len or len_k
//    const int   stride_c = attn_batches *k_seq_len; // k_seq_len or attn_batches * len_k

    // [len_k * head_dim] x [head_dim x attn_batches] -> [len_k x attn_batches]

//    Try to write in gemm_switch_fp32accum later
//    gemm_switch_fp32accum(   state,
//                             a_layout_n, //
//                             b_layout_n, //
//                             k_seq_len,  // m
//                             attn_batches,  // n
//                             head_dim,   // k
//                             scale,  // alpha
//                             static_cast<const half*>(pos.data_ptr()), // cm[len_k * head_dim * len_q]
//                             lda,
//                             batch_stride_a,
//                             static_cast<const half*>(rr_head_q_ptr),  // cm [head_dim * [attn_batches * len_q]]
//                             ldb, //
//                             batch_stride_b,
//                             beta_zero,
//                             static_cast<half*>(attn_scores_bd.data_ptr()),
//                             ldc,
//                             stride_c,
//                             batch_count_bd); // batch count

    attn_scores.view({sequences, heads, q_seq_len, k_seq_len}).masked_fill_(pad_mask,
                                                                          -std::numeric_limits<float>::infinity());

    bool softmax_success = false;

    if (is_training) {
        softmax_success = dispatch_softmax_dropout<half, half, float>(
                             reinterpret_cast<half*>(dropout_results_ptr),
                             reinterpret_cast<half*>(softmax_results_ptr),
                             reinterpret_cast<uint8_t*>(dropout_mask.data_ptr<uint8_t>()),
                             reinterpret_cast<const half*>(attn_scores_ptr),
                             dropout_elems,
                             k_seq_len,
                             k_seq_len,
                             attn_batches*q_seq_len,
                             (1.0f - dropout_prob),
                             stream);
    } else {
        softmax_success = dispatch_softmax<half, half, float>(
                             reinterpret_cast<half*>(dropout_results_ptr),
                             reinterpret_cast<const half*>(attn_scores_ptr),
                             dropout_elems,
                             k_seq_len,
                             k_seq_len,
                             attn_batches*q_seq_len,
                             stream);

        softmax_results.copy_(dropout_results);
    }

    assert(softmax_success);
//
//  v              : [len_k x attn_batches x head_dim] -> colm [head_dim x attn_batches) x len_k]
//  dropout results: [attn_batches x len_q x len_k] -> colm [len_k x [len_q x attn_batches]]
//  [head_dim x len_k] x [len_k x len_q] -> [head_dim x len_q] x
//
    // TODO: learning this function
    gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_n,
                             head_dim,
                             q_seq_len,
                             k_seq_len,
                             alpha,
                             static_cast<const half*>(v_lin_results_ptr),
                             lead_dim_qkv, // lda
                             batch_stride_qkv, // stride a
                             static_cast<const half*>(dropout_results.data_ptr()) ,
                             k_seq_len, // ldb
                             k_seq_len*q_seq_len, // strideb
                             beta_zero,
                             static_cast<half*>(matmul2_results.data_ptr()),
                             head_dim*attn_batches, // ldc
                             head_dim,  // c stride
                             attn_batches);
//
    outputs.copy_(output_biases);
//
    // Output Linear
    THCublasCheck(cublasGemmEx(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            embed_dim,
                            batches,
                            embed_dim,
                            static_cast<const void*>(&alpha),
                            static_cast<const void*>(output_weights.data_ptr()),
                            CUDA_R_16F,
                            embed_dim,
                            static_cast<const void*>(matmul2_results.data_ptr()),
                            CUDA_R_16F,
                            embed_dim,
                            static_cast<const void*>(&beta_one),
                            static_cast<void*>(outputs.data_ptr()),
                            CUDA_R_16F,
                            embed_dim,
                            CUDA_R_32F,
                            //CUBLAS_GEMM_ALGO1_TENSOR_OP));
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    return {
        input_lin_results,
        rr_head_q, rw_head_q,
        softmax_results, dropout_results, dropout_mask,
        matmul2_results,
        outputs
    };

}

std::vector<torch::Tensor> bwd_cuda(
                               int                  heads,
                               torch::Tensor const& output_grads,
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& softmax_results,
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& rw_head_q,
                               torch::Tensor const& rr_head_q,
                               torch::Tensor const& inputs,
                               torch::Tensor const& pos,
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
    const int   embed_dim         = inputs.size(2);
    const int   sequences         = inputs.size(1);
    const int   q_seq_len         = inputs.size(0);
    const int   k_seq_len         = q_seq_len;
    const int   batches         = sequences * q_seq_len;
    const int   head_dim          = embed_dim / heads;
    const int   output_lin_dim  = 3 * embed_dim;
    const int   attn_batches      = heads * sequences;
    const int   lead_dim_qkv        = attn_batches * 3 * head_dim;
    const int   batch_stride_qkv   = 3 * head_dim;
    const int   lead_dim        = attn_batches * head_dim;
    const int   batch_stride   = head_dim;
    //  const int   dropout_elems     = attn_batches * q_seq_len * k_seq_len;
    const float alpha             = 1.0;
    const float beta              = 0.0;
    const float scale             = 1.0 / sqrt(static_cast<float>(head_dim));

    // TODO: Streams can be used in Backprop but I haven't added more than one
    // in my first attempt to create the code
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    // Output Tensor Allocations
    torch::Tensor input_grads         = torch::empty_like(inputs);
    torch::Tensor pos_grads           = torch::empty_like(pos);
    torch::Tensor input_weight_grads  = torch::empty_like(input_weights);
    torch::Tensor output_weight_grads = torch::empty_like(output_weights);
    // Intermediate Tensor Allocations
    at::Tensor output_lin_grads       = torch::empty_like(matmul2_results);
    at::Tensor matmul2_grads          = torch::empty_like(dropout_results);
    at::Tensor input_lin_output_grads = torch::empty_like(input_lin_results);

//    void* rw_head_q_ptr   = static_cast<void*>(rw_head_q.view({q_seq_len, sequences, embed_dim}).data_ptr());
    void* rw_head_q_ptr   = static_cast<void*>(rw_head_q.data_ptr());

    auto q_lin_results_ptr = static_cast<half*>(input_lin_results.data_ptr());
    auto k_lin_results_ptr = static_cast<half*>(input_lin_results.data_ptr()) + head_dim;
    auto v_lin_results_ptr = static_cast<half*>(input_lin_results.data_ptr()) + 2*head_dim;

    auto q_lin_grads_ptr = static_cast<half*>(input_lin_output_grads.data_ptr());
    auto k_lin_grads_ptr = static_cast<half*>(input_lin_output_grads.data_ptr()) + head_dim;
    auto v_lin_grads_ptr = static_cast<half*>(input_lin_output_grads.data_ptr()) + 2*head_dim;

    // need a tensor at this position
    torch::Tensor queries_grads = input_lin_output_grads.view({q_seq_len, sequences*heads,
                                                  3, head_dim}).index({Slice(), Slice(), 0, Slice()});

//    torch::Tensor key = input_lin_results.view({q_seq_len, sequences*heads,
//                                                  3, head_dim}).index({Slice(), Slice(), 1, Slice()});

    torch::Tensor queries_grads_bd = torch::empty_like(queries_grads);
    char a_layout_n{'n'};
    char a_layout_t{'t'};
    char b_layout_n{'n'};
    char b_layout_t{'t'};

    THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Output Linear Dgrad
    THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             embed_dim,
                             batches,
                             embed_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(output_weights.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(output_grads.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(&beta),
                             static_cast<void*>(output_lin_grads.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Output Linear Wgrad
    THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             embed_dim,
                             embed_dim,
                             batches,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(matmul2_results.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(output_grads.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(&beta),
                             static_cast<void*>(output_weight_grads.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    auto  output_bias_grads = output_grads.view({-1, embed_dim}) .sum(0, false);
    // MatMul2 Dgrad1
    gemm_switch_fp32accum(     state,
                             a_layout_t,
                             b_layout_n,
                             k_seq_len,
                             q_seq_len,
                             head_dim,
                             alpha,
                             static_cast<const half*>(v_lin_results_ptr),
                             lead_dim_qkv,
                             batch_stride_qkv,
                             static_cast<const half*>(output_lin_grads.data_ptr()),
                             head_dim*attn_batches,
                             head_dim,
                             beta,
                             static_cast<half*>(matmul2_grads.data_ptr()),
                             k_seq_len,
                             k_seq_len*q_seq_len,
                             attn_batches);

    // Matmul2 Dgrad2
    gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_t,
                             head_dim,
                             k_seq_len,
                             q_seq_len,
                             alpha,
                             static_cast<const half*>(output_lin_grads.data_ptr()),
                             head_dim*attn_batches,
                             head_dim,
                             static_cast<const half*>(dropout_results.data_ptr()),
                             k_seq_len,
                             k_seq_len*q_seq_len,
                             beta,
                             v_lin_grads_ptr,
                             lead_dim_qkv,
                             batch_stride_qkv,
                             attn_batches);

    // bool softmax_success = false;

    dispatch_masked_scale_softmax_backward_recompute<half, half, float, false>(
                                 static_cast<half*>(matmul2_grads.data_ptr()),
                                 static_cast<half* const>(matmul2_grads.data_ptr()),
                                 reinterpret_cast<half const*>(softmax_results.data_ptr()),
                                 static_cast<uint8_t const*>(dropout_mask.data_ptr()),
                                 1.0/(1.0-dropout_prob),
                                 k_seq_len,
                                 k_seq_len,
                                 attn_batches*q_seq_len,
                                 stream);

    // after softmax we have attn_score_grads = matmul2_grads

//    auto matmul_ac_grads = matmul2_grads;

    // Matmul1 Dgrad1: first grads to the query_grad: multiply grads with keys -> queries_grad_ac
    gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_n,
                             head_dim,
                             q_seq_len,
                             k_seq_len,
                             scale,
                             k_lin_results_ptr,
                             lead_dim_qkv,
                             batch_stride_qkv,
                             static_cast<half*>(matmul2_grads.data_ptr()),
                             k_seq_len,
                             k_seq_len*q_seq_len,
                             beta,
                             q_lin_grads_ptr,  // queries_grads
                             lead_dim_qkv,
                             batch_stride_qkv,
                             attn_batches);

    auto r_w_bias_grads = queries_grads.view({q_seq_len * sequences, heads, head_dim}).sum(0, false);

    queries_grads_bd.baddbmm_(matmul2_grads.transpose(0, 1), pos, beta, scale);

    auto r_r_bias_grads = queries_grads_bd.view({q_seq_len * sequences, heads, head_dim}).sum(0, false);

    queries_grads.add_(queries_grads_bd);

    // backprop to get pos grads
    pos_grads.baddbmm_(matmul2_grads.transpose(0, 1).transpose(1, 2), rr_head_q, 0.0, scale);

    // Matmul1 Dgrad2
    gemm_switch_fp32accum(     state,
                             a_layout_n,
                             b_layout_t,
                             head_dim,
                             k_seq_len,
                             q_seq_len,
                             scale,
                             static_cast<half*>(rw_head_q_ptr),
                             lead_dim,  // because rw_head_q is not a sub-mat
                             batch_stride,
                             static_cast<half*>(matmul2_grads.data_ptr()),
                             k_seq_len,
                             k_seq_len*q_seq_len,
                             beta,
                             k_lin_grads_ptr,
                             lead_dim_qkv,
                             batch_stride_qkv,
                             attn_batches);

    // Input Linear Dgrad
    THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             embed_dim,
                             batches,
                             output_lin_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(input_lin_output_grads.data_ptr()),
                             //static_cast<const void*>(q_lin_grads_ptr),
                             CUDA_R_16F,
                             output_lin_dim,
                             static_cast<const void*>(&beta),
                             static_cast<void*>(input_grads.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             CUDA_R_32F,
                             //CUBLAS_GEMM_ALGO10_TENSOR_OP));
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Input Linear Wgrad
    THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             embed_dim,
                             output_lin_dim,
                             batches,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(inputs.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(q_lin_grads_ptr),
                             CUDA_R_16F,
                             output_lin_dim,
                             static_cast<const void*>(&beta),
                             static_cast<void*>(input_weight_grads.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    auto  input_bias_grads = input_lin_output_grads.view({-1, output_lin_dim}).sum(0, false);
    THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    return  {
        input_grads, pos_grads,
        input_weight_grads,
        input_bias_grads,
        output_weight_grads,
        output_bias_grads,
        r_w_bias_grads, r_r_bias_grads
    };

}


}
}
}