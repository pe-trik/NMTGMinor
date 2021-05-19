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
#include "layer_norm.h"

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

namespace multihead_attn {
namespace encdec {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(
							   bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                   ) 
{
  const int   embed_dim         = inputs_q.size(2);
  const int   sequences         = inputs_q.size(1);
  const int   q_seq_len         = inputs_q.size(0);
  const int   k_seq_len         = inputs_kv.size(0);
  const int   batches_q         = sequences * q_seq_len;
  const int   batches_kv        = sequences * k_seq_len;
  const int   head_dim          = embed_dim / heads;
  const int   output_lin_q_dim  = embed_dim;
  const int   output_lin_kv_dim = 2 * embed_dim;
  const int   attn_batches      = heads * sequences;
  const int   lead_dim_q        = attn_batches * head_dim;
  const int   lead_dim_kv       = attn_batches * 2 *head_dim;
  const int   batch_stride_q    = head_dim;
  const int   batch_stride_kv   = 2 * head_dim;
  const int   dropout_elems     = attn_batches * q_seq_len * k_seq_len;
  const float alpha             = 1.0;
  const float beta              = 0.0;
  const float scale             = 1.0 / sqrt(static_cast<float>(head_dim));

//  printf("Input kernel sizes: %d %d %d \n",
//			inputs_kv.size(0), inputs_kv.size(1), inputs_kv.size(2));
 
  // There is no reason to use more than one stream as every kernel is 
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t   stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // 3 Intermediate Results + Output (Note: dropout intermediates are generated by ATen library code)
  auto act_options  = inputs_q.options().requires_grad(false);
  auto mask_options = act_options.dtype(torch::kUInt8);

  torch::Tensor input_lin_q_results  = torch::empty({q_seq_len, sequences, output_lin_q_dim},  act_options);
  torch::Tensor input_lin_kv_results = torch::empty({k_seq_len, sequences, output_lin_kv_dim}, act_options);
  torch::Tensor attn_scores          = torch::empty({attn_batches, q_seq_len, k_seq_len},      act_options);
  torch::Tensor softmax_results      = torch::empty({attn_batches, q_seq_len, k_seq_len},      act_options);
  torch::Tensor dropout_results      = torch::empty({attn_batches, q_seq_len, k_seq_len},      act_options);
  torch::Tensor dropout_mask         = torch::empty({attn_batches, q_seq_len, k_seq_len},      mask_options);
  torch::Tensor matmul2_results      = torch::empty({q_seq_len, attn_batches, head_dim},       act_options);
  torch::Tensor outputs              = torch::empty_like(inputs_q, act_options);

  // Input Linear Results Pointers to Q, K, and V of interviewed activations
  void* q_lin_results_ptr   = static_cast<void*>(input_lin_q_results.data_ptr());
  void* k_lin_results_ptr   = static_cast<void*>(input_lin_kv_results.data_ptr());
  void* v_lin_results_ptr   = static_cast<void*>(static_cast<half*>(input_lin_kv_results.data_ptr()) + head_dim);
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());
  void* dropout_results_ptr = static_cast<void*>(dropout_results.data_ptr());
  // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
  void* attn_scores_ptr     = static_cast<void*>(attn_scores.data_ptr());

  char a_layout_t{'t'};
  char a_layout_n{'n'};
  char b_layout_n{'n'};

  THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  // Input Linear Q Fwd
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_T, // A transpose
                             CUBLAS_OP_N, // B wo/ transpose
                             output_lin_q_dim, // embed_dim
                             batches_q,  // bsz x len_q
                             embed_dim,  // embed_dim
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights_q.data_ptr()), // weight emb_out x emb_in transposed
                             CUDA_R_16F,
                             embed_dim, // lda  so A has size [lda x m] -> [embed_dim x output_lin_q_dim]
                             static_cast<const void*>(inputs_q.data_ptr()), // input Q
                             CUDA_R_16F,
                             embed_dim, // ldb B has size [lda xn] -> [embed_dim x batches_q]
                             static_cast<const void*>(&beta), // beta
                             q_lin_results_ptr, // C -> emb * B
                             CUDA_R_16F,
                             output_lin_q_dim, // ldc C [lda x n] -> [embed_dim x batches_q]
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Input Linear KV Fwd
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_T, 
                             CUBLAS_OP_N,
                             output_lin_kv_dim, 
                             batches_kv, 
                             embed_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights_kv.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             static_cast<const void*>(inputs_kv.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim, 
                             static_cast<const void*>(&beta),
                             k_lin_results_ptr,
                             CUDA_R_16F, 
                             output_lin_kv_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // MatMul1 of Dot-Product Attention Plus scaling by 1/Sqrt(head size)
  gemm_switch_fp32accum(     state, 
                             a_layout_t, 
                             b_layout_n, 
                             k_seq_len, // m
                             q_seq_len, // n
                             head_dim, // k
                             scale, 
                             static_cast<const half*>(k_lin_results_ptr), 
                             lead_dim_kv, // lda
                             batch_stride_kv,  //strideA
                             static_cast<const half*>(q_lin_results_ptr),
                             lead_dim_q,  // ldb
                             batch_stride_q,  //strideB
                             beta, 
                             static_cast<half*>(attn_scores_ptr), // [attn_batches * len_q * len_k]
                             k_seq_len,  // ldc
                             k_seq_len*q_seq_len, // stride c
                             attn_batches); // p

  // need to call padding from torch interface here.
  attn_scores.view({sequences, heads, q_seq_len, k_seq_len}).masked_fill_(pad_mask,
                                                                          -std::numeric_limits<float>::infinity());

  attn_scores.view({sequences*heads, q_seq_len, k_seq_len});
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

  // Matmul2
  // matrix kv has size len_k * batch_size * (2 * heads * head_dim)
  // dropout results [bsz*heads, len_q, len_k]
  // matmul2_results is [len_q x attn_batches x head_dim]
  gemm_switch_fp32accum(     state, 
                             a_layout_n, 
                             b_layout_n, 
                             head_dim,  // m
                             q_seq_len,  // n
                             k_seq_len,  // k
                             alpha, 
                             static_cast<const half*>(v_lin_results_ptr), // A_i [head_dimxk_seq_len]
                             lead_dim_kv,  // attn_batches * 2 *head_dim
                             batch_stride_kv,  // stride = 2 * head_dim
                             static_cast<const half*>(dropout_results.data_ptr()), // B_i [k_seq_len x q_seq_len]
                             k_seq_len, // lead_dim
                             k_seq_len*q_seq_len,  // stride
                             beta, 
                             static_cast<half*>(matmul2_results.data_ptr()), 
                             head_dim*attn_batches, // ldc
                             head_dim,  // stride c
                             attn_batches); //p

  // Output Linear
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_T, 
                             CUBLAS_OP_N,
                             embed_dim, 
                             batches_q, 
                             embed_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(output_weights.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             static_cast<const void*>(matmul2_results.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim, 
                             static_cast<const void*>(&beta),
                             static_cast<void*>(outputs.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             CUDA_R_32F,
                             //CUBLAS_GEMM_ALGO1_TENSOR_OP));
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return { 
           input_lin_q_results, 
           input_lin_kv_results,
           softmax_results,
           dropout_results,
           dropout_mask, 
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
                               torch::Tensor const& input_lin_q_results,
                               torch::Tensor const& input_lin_kv_results,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  ) 
{
  const int   embed_dim         = inputs_q.size(2);
  const int   sequences         = inputs_q.size(1);
  const int   q_seq_len         = inputs_q.size(0);
  const int   k_seq_len         = inputs_kv.size(0);
  const int   batches_q         = sequences * q_seq_len;
  const int   batches_kv        = sequences * k_seq_len;
  const int   head_dim          = embed_dim / heads;
  const int   output_lin_q_dim  = embed_dim;
  const int   output_lin_kv_dim = 2 * embed_dim;
  const int   attn_batches      = heads * sequences;
  const int   lead_dim_q        = attn_batches * head_dim;
  const int   lead_dim_kv       = attn_batches * 2 *head_dim;
  const int   batch_stride_q    = head_dim;
  const int   batch_stride_kv   = 2 * head_dim;
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
  torch::Tensor input_q_grads          = torch::empty_like(inputs_q);
  torch::Tensor input_kv_grads         = torch::empty_like(inputs_kv);
  torch::Tensor input_weight_q_grads   = torch::empty_like(input_weights_q);
  torch::Tensor input_weight_kv_grads  = torch::empty_like(input_weights_kv);
  torch::Tensor output_weight_grads    = torch::empty_like(output_weights);
  // Intermediate Tensor Allocations
  at::Tensor output_lin_grads          = torch::empty_like(matmul2_results);
  at::Tensor matmul2_grads             = torch::empty_like(softmax_results);
  at::Tensor input_lin_q_output_grads  = torch::empty_like(input_lin_q_results);
  at::Tensor input_lin_kv_output_grads = torch::empty_like(input_lin_kv_results);
 
  auto q_lin_results_ptr = static_cast<half*>(input_lin_q_results.data_ptr());
  auto k_lin_results_ptr = static_cast<half*>(input_lin_kv_results.data_ptr());
  auto v_lin_results_ptr = static_cast<half*>(input_lin_kv_results.data_ptr()) + head_dim;
  
  auto q_lin_grads_ptr   = static_cast<half*>(input_lin_q_output_grads.data_ptr());
  auto k_lin_grads_ptr   = static_cast<half*>(input_lin_kv_output_grads.data_ptr());
  auto v_lin_grads_ptr   = static_cast<half*>(input_lin_kv_output_grads.data_ptr()) + head_dim;

  char a_layout_n{'n'};
  char a_layout_t{'t'};
  char b_layout_n{'n'};
  char b_layout_t{'t'}; 
  
  THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
 
  // Output Linear Dgrad
  // C = alpha * op(A) op(B) + BetaC
  // op(A): mxk, op(B): kxn C: mxn
  THCublasCheck(cublasGemmEx(handle,  //
                             CUBLAS_OP_N, // no transpose
                             CUBLAS_OP_N, // no transpose
                             embed_dim, // m
                             batches_q, // n = bsz * len_q
                             embed_dim, // k
                             static_cast<const void*>(&alpha),  // alpha = 1.0
                             static_cast<const void*>(output_weights.data_ptr()), // A mxk
                             CUDA_R_16F, // data type
                             embed_dim,  // leading dimension of A (embed dim) (the rows)
                             static_cast<const void*>(output_grads.data_ptr()), // B kxn
                             CUDA_R_16F, // data type
                             embed_dim,  // leading dimension of B (embed dim)
                             static_cast<const void*>(&beta), // beta
                             static_cast<void*>(output_lin_grads.data_ptr()), // C mxn
                             CUDA_R_16F,  // data type
                             embed_dim, // ldc
                             CUDA_R_32F, // compute type
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 
  // Output Linear Wgrad
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N, 
                             CUBLAS_OP_T,
                             embed_dim, 
                             embed_dim,
                             batches_q, 
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
  
  // MatMul2 Dgrad1
  gemm_switch_fp32accum(     state, 
                             a_layout_t, 
                             b_layout_n, 
                             k_seq_len,
                             q_seq_len,
                             head_dim,
                             alpha, 
                             static_cast<const half*>(v_lin_results_ptr),
                             lead_dim_kv,
                             batch_stride_kv, // 2 * head_dim
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
                             lead_dim_kv, 
                             batch_stride_kv, 
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

  // Matmul1 Dgrad1
  gemm_switch_fp32accum(     state, 
                             a_layout_n, 
                             b_layout_n, 
                             head_dim, 
                             q_seq_len, 
                             k_seq_len, 
                             scale, 
                             k_lin_results_ptr, 
                             lead_dim_kv, 
                             batch_stride_kv, 
                             static_cast<half*>(matmul2_grads.data_ptr()),
                             k_seq_len, 
                             k_seq_len*q_seq_len, 
                             beta, 
                             q_lin_grads_ptr, 
                             lead_dim_q, 
                             batch_stride_q, 
                             attn_batches);
  
  // Matmul1 Dgrad2
  gemm_switch_fp32accum(     state, 
                             a_layout_n, 
                             b_layout_t, 
                             head_dim, 
                             k_seq_len, 
                             q_seq_len, 
                             scale, 
                             q_lin_results_ptr, 
                             lead_dim_q, 
                             batch_stride_q, 
                             static_cast<half*>(matmul2_grads.data_ptr()),
                             k_seq_len, 
                             k_seq_len*q_seq_len, 
                             beta, 
                             k_lin_grads_ptr, 
                             lead_dim_kv, 
                             batch_stride_kv, 
                             attn_batches);

  // Input Linear Q Dgrad  
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N, 
                             CUBLAS_OP_N,
                             embed_dim,
                             batches_q, 
                             output_lin_q_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights_q.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             static_cast<const void*>(q_lin_grads_ptr),
                             CUDA_R_16F, 
                             output_lin_q_dim, 
                             static_cast<const void*>(&beta),
                             static_cast<void*>(input_q_grads.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             CUDA_R_32F,
                             //CUBLAS_GEMM_ALGO10_TENSOR_OP));
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  // Input Linear Q Wgrad  
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N, 
                             CUBLAS_OP_T,
                             embed_dim, 
                             output_lin_q_dim,
                             batches_q, 
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(inputs_q.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(q_lin_grads_ptr),
                             CUDA_R_16F,
                             output_lin_q_dim,
                             static_cast<const void*>(&beta),
                             static_cast<void*>(input_weight_q_grads.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  // Input Linear KV Dgrad  
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N, 
                             CUBLAS_OP_N,
                             embed_dim,
                             batches_kv, 
                             output_lin_kv_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights_kv.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             static_cast<const void*>(k_lin_grads_ptr),
                             CUDA_R_16F, 
                             output_lin_kv_dim, 
                             static_cast<const void*>(&beta),
                             static_cast<void*>(input_kv_grads.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             CUDA_R_32F,
                             //CUBLAS_GEMM_ALGO10_TENSOR_OP));
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  
  // Input Linear KV Wgrad  
  THCublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N, 
                             CUBLAS_OP_T,
                             embed_dim, 
                             output_lin_kv_dim,
                             batches_kv, 
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(inputs_kv.data_ptr()),
                             CUDA_R_16F,
                             embed_dim,
                             static_cast<const void*>(k_lin_grads_ptr),
                             CUDA_R_16F,
                             output_lin_kv_dim,
                             static_cast<const void*>(&beta),
                             static_cast<void*>(input_weight_kv_grads.data_ptr()),
                             CUDA_R_16F, 
                             embed_dim,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return { 
           input_q_grads, 
           input_kv_grads, 
           input_weight_q_grads, 
           input_weight_kv_grads, 
           output_weight_grads
         };
}

} // end namespace cublas_gemmex
} // end namespace encdec 
} // end namespace multihead_attn
