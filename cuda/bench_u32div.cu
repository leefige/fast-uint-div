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

using my_clock_t = clock_t;

__device__ __forceinline__ my_clock_t get_clock() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%clock;\n" : "=r"(ret));
  return ret;
}

namespace impl {

struct KernelImpl {
  template <typename Func>
  __device__ __forceinline__ void self_op(uint32_t &x, const U32Div &div,
                                          Func &&func) const {
    x = func(x, div);
    return;
  }

  template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1, typename Func>
  __device__ __forceinline__ void Run(my_clock_t *cycles, uint32_t *out,
                                      const uint32_t *dividends,
                                      const U32Div &div) const {
    static_assert(UNROLL_0 < UNROLL_1,
                  "UNROLL_0 should be smaller than UNROLL_1");
    uint32_t regs[N_REGS];

    // load
#pragma unroll
    for (int i = 0; i < N_REGS; ++i) {
      regs[i] = dividends[threadIdx.x + i * BLOCK];
    }

    // warm-up
#pragma unroll
    for (int i = 0; i < N_REGS; ++i) {
      self_op(regs[i], div, Func());
    }

    my_clock_t start, end;

    // first pass
    start = get_clock();
#pragma unroll
    for (int i = 0; i < UNROLL_0; ++i) {
#pragma unroll
      for (int i = 0; i < N_REGS; ++i) {
        self_op(regs[i], div, Func());
      }
    }
    end = get_clock();
    my_clock_t cycles_0 = end - start;

    // second pass
    start = get_clock();
#pragma unroll
    for (int i = 0; i < UNROLL_1; ++i) {
#pragma unroll
      for (int i = 0; i < N_REGS; ++i) {
        self_op(regs[i], div, Func());
      }
    }
    end = get_clock();
    my_clock_t cycles_1 = end - start;

    // time it
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *cycles = cycles_1 - cycles_0;
    }

    // store
#pragma unroll
    for (int i = 0; i < N_REGS; ++i) {
      out[threadIdx.x + i * BLOCK] = regs[i];
    }

    return;
  }
};

template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1,
          typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_reference(my_clock_t *cycles, uint32_t *out,
                     const uint32_t *dividends, const U32Div div) {
  // use ptx to prevent compiler optimization
  KernelImpl().template Run<BLOCK, N_REGS, UNROLL_0, UNROLL_1, DivideRefPtx>(
      cycles, out, dividends, div);
}

template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1,
          typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div(my_clock_t *cycles, uint32_t *out, const uint32_t *dividends,
               const U32Div div) {
  KernelImpl().template Run<BLOCK, N_REGS, UNROLL_0, UNROLL_1, Divide>(
      cycles, out, dividends, div);
}

template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1,
          typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div_bounded(my_clock_t *cycles, uint32_t *out,
                       const uint32_t *dividends, const U32Div div) {
  KernelImpl().template Run<BLOCK, N_REGS, UNROLL_0, UNROLL_1, DivideBounded>(
      cycles, out, dividends, div);
}

} // namespace impl

template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1>
struct KernelReference {
  void operator()(my_clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_reference<BLOCK, N_REGS, UNROLL_0, UNROLL_1, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1>
struct KernelDiv {
  void operator()(my_clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_div<BLOCK, N_REGS, UNROLL_0, UNROLL_1, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int BLOCK, int N_REGS, int UNROLL_0, int UNROLL_1>
struct KernelDivBounded {
  void operator()(my_clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_div_bounded<BLOCK, N_REGS, UNROLL_0, UNROLL_1,
                             impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int N_REGS,
          template <int /* BLOCK */, int /* N_REGS */, int /* UNROLL_0 */,
                    int /* UNROLL_1 */> typename Kernel>
class Test {
  static constexpr int N_ROUNDS = 50;
  static constexpr int UNROLL_0 = 64;
  static constexpr int UNROLL_1 = 96;
  static constexpr int LENGTH = N_REGS * CTA_SIZE;
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
           KernelReference<CTA_SIZE, N_REGS, UNROLL_0, UNROLL_1>());

    /* run target */
    run_it(total_time_fast, cycles_fast, target_d, "target",
           Kernel<CTA_SIZE, N_REGS, UNROLL_0, UNROLL_1>());

    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("1 warp, %02d regs per thread, #invocation: %d, "
           "reference: %lld cycles,\ttarget: %lld cycles\n",
           N_REGS, UNROLL_1 - UNROLL_0, static_cast<long long>(total_time_slow),
           static_cast<long long>(total_time_fast));
    return;
  }

private:
  void setup() {
    CHECK_CUDA(cudaMalloc(&n_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&ref_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&target_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&cycles_fast, sizeof(my_clock_t)));
    CHECK_CUDA(cudaMalloc(&cycles_slow, sizeof(my_clock_t)));
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
  void run_it(my_clock_t &duration, my_clock_t *cycles, uint32_t *data_device,
              const char *name, Func &&func) {
    for (int i = 0; i < N_ROUNDS; ++i) {
      func(cycles, data_device, n_d, div, CTA_COUNT, stream);
      CHECK_KERNEL();
    }
    CHECK_CUDA(cudaMemcpyAsync(&duration, cycles, sizeof(my_clock_t),
                               cudaMemcpyDeviceToHost, stream));

    return;
  }

  uint32_t *n_d;
  uint32_t *ref_d;
  uint32_t *target_d;

  my_clock_t *cycles_slow;
  my_clock_t *cycles_fast;

  cudaStream_t stream;

  U32Div div;
  my_clock_t total_time_slow;
  my_clock_t total_time_fast;
  bool large_n;
};

template <int N_REGS>
using TestDivBounded = Test<N_REGS, KernelDivBounded>;
template <int N_REGS>
using TestDiv = Test<N_REGS, KernelDiv>;

int main() {
  constexpr int N_CASES = 3;

  srand((unsigned)time(nullptr));

  puts("DivBounded");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = rand() + 1U;
    TestDivBounded<1> test(d);
    test.Run();
  }

  puts("\nDivBounded");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = rand() + 1U;
    TestDivBounded<4> test(d);
    test.Run();
  }

  puts("\nDivBounded");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = rand() + 1U;
    TestDivBounded<16> test(d);
    test.Run();
  }

  puts("\nDivBounded");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = rand() + 1U;
    TestDivBounded<64> test(d);
    test.Run();
  }

  puts("\nDiv");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = (1U << 31);
    TestDiv<1> test(d);
    test.Run();
  }

  puts("\nDiv");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = (1U << 31);
    TestDiv<4> test(d);
    test.Run();
  }

  puts("\nDiv");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = (1U << 31);
    TestDiv<16> test(d);
    test.Run();
  }

  puts("\nDiv");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = (1U << 31);
    TestDiv<64> test(d);
    test.Run();
  }

  return 0;
}
