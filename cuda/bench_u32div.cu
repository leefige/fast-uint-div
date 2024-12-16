/*!
 * Copyright (c) 2024 Yifei Li
 * SPDX-License-Identifier: MIT
 *
 * The implementation of constexpr_for is credit to Philip Trettner.
 */

#include "utils.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <type_traits>

// We use only 1 warp for benchmark.
static constexpr int WARP_SIZE = 32;
static constexpr int CTA_SIZE = WARP_SIZE;
static constexpr int CTA_COUNT = 1;

/* Use the difference of two unrolling to represent the cycles of UNROLL_1 -
 * UNROLL_0 invocations, eliminating the tailing effect and common overhead.
 */
static constexpr int UNROLL_0 = 32;
static constexpr int UNROLL_1 = 64;

namespace impl {

__device__ __forceinline__ clock_t get_clock() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%clock;\n" : "=r"(ret));
  return ret;
}

template <int BLOCK, int N_REGS, int UNROLL, typename Func>
__global__ void __launch_bounds__(BLOCK, 1)
    bench_kernel(clock_t *cycles, uint32_t *out, const U32Div div) {
  uint32_t regs[N_REGS];

  // init
#pragma unroll
  for (int i = 0; i < N_REGS; ++i) {
    regs[i] = threadIdx.x * BLOCK + i;
  }

  // self-MA to ensure regs ready
#pragma unroll
  for (int i = 0; i < N_REGS; ++i) {
    regs[i] = regs[i] * regs[i] + regs[i];
  }

  // latency for UNROLL times
  Func func;
  clock_t start = get_clock();
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
    for (int i = 0; i < N_REGS; ++i) {
      regs[i] = func(regs[i], div);
    }
  }
  clock_t end = get_clock();
  *cycles = end - start;

  // store
#pragma unroll
  for (int i = 0; i < N_REGS; ++i) {
    out[threadIdx.x + i * BLOCK] = regs[i];
  }
  return;
}

} // namespace impl

template <int BLOCK, int N_REGS, int UNROLL, typename DivideFunc>
struct DivideKernel {
  static void Run(clock_t *cycles, uint32_t *out, const U32Div &div,
                  int n_blocks, cudaStream_t stream) {
    impl::bench_kernel<BLOCK, N_REGS, UNROLL, DivideFunc>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, div);
  }
};

template <int BLOCK, int N_REGS, int UNROLL>
using KernelReference = DivideKernel<BLOCK, N_REGS, UNROLL, impl::DivideRef>;

template <int BLOCK, int N_REGS, int UNROLL>
using KernelDiv = DivideKernel<BLOCK, N_REGS, UNROLL, impl::Divide>;

template <int BLOCK, int N_REGS, int UNROLL>
using KernelDivBounded =
    DivideKernel<BLOCK, N_REGS, UNROLL, impl::DivideBounded>;

template <int N_REGS, int UNROLL_0, int UNROLL_1,
          template <int /* BLOCK */, int /* N_REGS */,
                    int /* UNROLL */> typename Kernel>
class Test {
  /* Launch kernel for multiple rounds to:
   *  1. warm-up loading module (device code);
   *  2. eliminate potential overhead of context creation.
   */
  static constexpr int N_ROUNDS = 3;
  static constexpr int LENGTH = N_REGS * CTA_SIZE;

public:
  explicit Test(uint32_t d_)
      : div(d_), cycles_slow_0(0), cycles_slow_1(0), cycles_fast_0(0),
        cycles_fast_1(0) {
    setup();
  }
  virtual ~Test() { cleanup(); }

  void Run() {
    /* run reference */
    run_async<KernelReference>(cycles_slow_0, cycles_slow_1);

    /* run target */
    run_async<Kernel>(cycles_fast_0, cycles_fast_1);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    clock_t total_time_slow = cycles_slow_1 - cycles_slow_0;
    clock_t total_time_fast = cycles_fast_1 - cycles_fast_0;

    printf("%d warp, %d regs per thread, #invocation: %d, "
           "reference: %lld cycles,\ttarget: %lld cycles,\tspeedup: %.2lf\n",
           CTA_SIZE / WARP_SIZE, N_REGS, UNROLL_1 - UNROLL_0,
           static_cast<long long>(total_time_slow),
           static_cast<long long>(total_time_fast),
           static_cast<double>(total_time_slow) /
               static_cast<double>(total_time_fast));
    return;
  }

private:
  void setup() {
    CHECK_CUDA(cudaMalloc(&out_d, LENGTH * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&cycles_unroll_1, CTA_SIZE * sizeof(clock_t)));
    CHECK_CUDA(cudaMalloc(&cycles_unroll_0, CTA_SIZE * sizeof(clock_t)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    return;
  }

  void cleanup() {
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(cycles_unroll_1));
    CHECK_CUDA(cudaFree(cycles_unroll_0));
    CHECK_CUDA(cudaFree(out_d));
    return;
  }

  template <template <int /* BLOCK */, int /* N_REGS */,
                      int /* UNROLL */> typename Kern>
  void run_async(clock_t &cycles_h_0, clock_t &cycles_h_1) {
    for (int i = 0; i < N_ROUNDS; ++i) {
      Kern<CTA_SIZE, N_REGS, UNROLL_0>::Run(cycles_unroll_0, out_d, div,
                                            CTA_COUNT, stream);
      CHECK_KERNEL();
    }

    for (int i = 0; i < N_ROUNDS; ++i) {
      Kern<CTA_SIZE, N_REGS, UNROLL_1>::Run(cycles_unroll_1, out_d, div,
                                            CTA_COUNT, stream);
      CHECK_KERNEL();
    }

    // use the cycles of thread 0
    CHECK_CUDA(cudaMemcpyAsync(&cycles_h_0, cycles_unroll_0, sizeof(clock_t),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&cycles_h_1, cycles_unroll_1, sizeof(clock_t),
                               cudaMemcpyDeviceToHost, stream));
    return;
  }

  U32Div div;
  clock_t cycles_slow_0;
  clock_t cycles_slow_1;
  clock_t cycles_fast_0;
  clock_t cycles_fast_1;

  uint32_t *out_d;
  clock_t *cycles_unroll_0;
  clock_t *cycles_unroll_1;

  cudaStream_t stream;
};

template <int N_REGS>
using TestDivBounded = Test<N_REGS, UNROLL_0, UNROLL_1, KernelDivBounded>;
template <int N_REGS>
using TestDiv = Test<N_REGS, UNROLL_0, UNROLL_1, KernelDiv>;

/* Credit to Philip Trettner:
 * https://artificial-mind.net/blog/2020/10/31/constexpr-for
 */
template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

int main() {
  constexpr int MAX_ILP_REGS = 8;

  srand((unsigned)time(nullptr));
  uint32_t d = rand() + 1U;

  puts("DivBounded");
  constexpr_for<1, MAX_ILP_REGS + 1, 1>([&](auto i) {
    TestDivBounded<i> test(d);
    test.Run();
  });

  puts("\nDiv");
  constexpr_for<1, MAX_ILP_REGS + 1, 1>([&](auto i) {
    TestDiv<i> test(d);
    test.Run();
  });

  return 0;
}
