/*!
 * Copyright (c) 2024 Yifei Li
 * SPDX-License-Identifier: MIT
 */

#include "utils.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// we use only 1 warp for benchmark
static constexpr int WARP_SIZE = 32;
static constexpr int CTA_SIZE = WARP_SIZE;

namespace impl {

// one thread computes all elements
struct KernelImpl {
  template <typename Func>
  __device__ __forceinline__ void self_op(uint32_t &x, const U32Div &div,
                                          Func &&func) const {
    x = func(x, div);
    return;
  }

  template <int BLOCK, int UNROLL_0, int UNROLL_1, typename Func>
  __device__ __forceinline__ void Run(clock_t *cycles, uint32_t *out,
                                      const uint32_t *dividends,
                                      const U32Div &div) const {
    static_assert(UNROLL_0 < UNROLL_1,
                  "UNROLL_0 should be smaller than UNROLL_1");
    uint32_t regs[UNROLL_1];

    // load
#pragma unroll
    for (int i = 0; i < UNROLL_1; ++i) {
      regs[i] = dividends[threadIdx.x + i * BLOCK];
    }

// warm-up
#pragma unroll
    for (int i = 0; i < UNROLL_1; ++i) {
      self_op(regs[i], div, Func());
    }

    clock_t start, end;

    // first pass
    start = clock();
#pragma unroll
    for (int i = 0; i < UNROLL_1; ++i) {
      self_op(regs[i], div, Func());
    }
    end = clock();
    clock_t cycles_1 = end - start;

    // second pass
    start = clock();
#pragma unroll
    for (int i = 0; i < UNROLL_0; ++i) {
      self_op(regs[i], div, Func());
    }
    end = clock();
    clock_t cycles_0 = end - start;

    // time it
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *cycles = cycles_1 - cycles_0;
    }

// store
#pragma unroll
    for (int i = 0; i < UNROLL_1; ++i) {
      out[threadIdx.x + i * BLOCK] = regs[i];
    }

    return;
  }
};

template <int BLOCK, int UNROLL_0, int UNROLL_1, typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_reference(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                     const U32Div div) {
  KernelImpl().template Run<BLOCK, UNROLL_0, UNROLL_1, DivideRef>(
      cycles, out, dividends, div);
}

template <int BLOCK, int UNROLL_0, int UNROLL_1, typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
               const U32Div div) {
  KernelImpl().template Run<BLOCK, UNROLL_0, UNROLL_1, Divide>(cycles, out,
                                                               dividends, div);
}

template <int BLOCK, int UNROLL_0, int UNROLL_1, typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div_bounded(clock_t *cycles, uint32_t *out,
                       const uint32_t *dividends, const U32Div div) {
  KernelImpl().template Run<BLOCK, UNROLL_0, UNROLL_1, DivideBounded>(
      cycles, out, dividends, div);
}

} // namespace impl

template <int BLOCK, int UNROLL_0, int UNROLL_1>
struct KernelReference {
  void operator()(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_reference<BLOCK, UNROLL_0, UNROLL_1, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int BLOCK, int UNROLL_0, int UNROLL_1>
struct KernelDiv {
  void operator()(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_div<BLOCK, UNROLL_0, UNROLL_1, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int BLOCK, int UNROLL_0, int UNROLL_1>
struct KernelDivBounded {
  void operator()(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_div_bounded<BLOCK, UNROLL_0, UNROLL_1, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <template <int BLOCK, int UNROLL_0, int UNROLL_1> typename Kernel>
class Test {
  static constexpr int N_ROUNDS = 50;
  static constexpr int UNROLL_0 = 64;
  static constexpr int UNROLL_1 = 96;
  static constexpr int LENGTH = UNROLL_1 * CTA_SIZE;
  // we use only 1 block for benchmark
  static constexpr int CTA_COUNT = 1;

public:
  explicit Test(uint32_t d_) : div(d_), total_time_slow(0), total_time_fast(0) {
    setup();
  }
  virtual ~Test() { cleanup(); }

  void Run() {
    CHECK_CUDA(cudaMemsetAsync(n_d, -1, LENGTH * sizeof(uint32_t), stream));

    /* run reference */
    run_it(total_time_slow, cycles_slow, ref_d, "reference",
           KernelReference<CTA_SIZE, UNROLL_0, UNROLL_1>());

    /* run target */
    run_it(total_time_fast, cycles_fast, target_d, "target",
           Kernel<CTA_SIZE, UNROLL_0, UNROLL_1>());

    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("1 warp, #invocation: %d,\treference: %lld cycles,\ttarget: "
           "%lld cycles\n",
           UNROLL_1 - UNROLL_0, static_cast<long long>(total_time_slow),
           static_cast<long long>(total_time_fast));
    return;
  }

private:
  void setup() {
    CHECK_CUDA(cudaMalloc(&n_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&ref_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&target_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&cycles_fast, sizeof(clock_t)));
    CHECK_CUDA(cudaMalloc(&cycles_slow, sizeof(clock_t)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    return;
  }

  void cleanup() {
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(cycles_fast));
    CHECK_CUDA(cudaFree(cycles_slow));
    CHECK_CUDA(cudaFree(target_d));
    CHECK_CUDA(cudaFree(ref_d));
    CHECK_CUDA(cudaFree(n_d));
    return;
  }

  template <typename Func>
  void run_it(clock_t &duration, clock_t *cycles, uint32_t *data_device,
              const char *name, Func &&func) {
    for (int i = 0; i < N_ROUNDS; ++i) {
      func(cycles, data_device, n_d, div, CTA_COUNT, stream);
      CHECK_KERNEL();
    }
    CHECK_CUDA(cudaMemcpyAsync(&duration, cycles, sizeof(clock_t),
                               cudaMemcpyDeviceToHost, stream));

    return;
  }

  uint32_t *n_d;
  uint32_t *ref_d;
  uint32_t *target_d;

  clock_t *cycles_slow;
  clock_t *cycles_fast;

  cudaStream_t stream;

  U32Div div;
  clock_t total_time_slow;
  clock_t total_time_fast;
  bool large_n;
};

using TestDivBounded = Test<KernelDivBounded>;
using TestDiv = Test<KernelDiv>;

int main() {
  constexpr int N_CASES = 5;

  srand((unsigned)time(nullptr));

  puts("DivBounded");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = rand() + 1U;
    TestDivBounded test(d);
    test.Run();
  }

  puts("\nDiv");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = (1U << 31);
    TestDiv test(d);
    test.Run();
  }

  return 0;
}
