/*!
 * Copyright (c) 2024 Yifei Li
 * SPDX-License-Identifier: MIT
 */

#include "u32div.cuh"

#include <cstdint>

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("[CUDA Error] code %d at %s:%d: %s\n", err, __FILE__, __LINE__,   \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_KERNEL()                                                         \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      printf("[CUDA Error] code %d at %s:%d: %s\n", err, __FILE__, __LINE__,   \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace impl {

template <int N, typename T>
struct alignas(sizeof(uint4)) Vec {
  T data[N];
};

struct DivideRef {
  __device__ __forceinline__ uint32_t operator()(uint32_t n,
                                                 const U32Div &div) const {
    return n / div.GetD();
  }
};

struct DivideBounded {
  __device__ __forceinline__ uint32_t operator()(uint32_t n,
                                                 const U32Div &div) const {
    return div.DivBounded(n);
  }
};

struct Divide {
  __device__ __forceinline__ uint32_t operator()(uint32_t n,
                                                 const U32Div &div) const {
    return div.Div(n);
  }
};

} // namespace impl
