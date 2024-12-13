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

static constexpr int HOST_THREAD_COUNT = 16;
static constexpr int WARP_SIZE = 32;
// we use only 1 warp for benchmark
static constexpr int CTA_SIZE = WARP_SIZE;
static constexpr int TEST_COUNT = 1 << 24;
static_assert(TEST_COUNT % HOST_THREAD_COUNT == 0,
              "requires TEST_COUNT % HOST_THREAD_COUNT == 0");

namespace impl {

// one thread computes all elements
struct KernelImpl {
  template <int BLOCK, int UNROLL, typename Func>
  __device__ __forceinline__ void Run(clock_t *cycles, uint32_t *out,
                                      const uint32_t *dividends,
                                      const U32Div &div) const {
    constexpr int PACK = 16 / sizeof(uint32_t);
    using VecT = Vec<PACK, uint32_t>;
    static_assert(sizeof(VecT) == sizeof(uint4),
                  "assert sizeof(VecT) == sizeof(uint4)");
    static_assert(TEST_COUNT % (BLOCK * UNROLL * PACK) == 0,
                  "requires TEST_COUNT % (BLOCK * UNROLL * PACK) == 0");

    VecT v_in[UNROLL];
    const VecT *in_ptr = reinterpret_cast<const VecT *>(dividends) +
                         blockIdx.x * BLOCK * UNROLL + threadIdx.x;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      // load next vec
      v_in[i] = *(in_ptr + i * BLOCK);
    }

    VecT v_out[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      for (int k = 0; k < PACK; ++k) {
        v_out[i].data[k] = Func()(v_in[i].data[k], div);
      }
    }

    VecT *out_ptr = reinterpret_cast<VecT *>(out) +
                    blockIdx.x * BLOCK * UNROLL + threadIdx.x;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      *(out_ptr + i * BLOCK) = v_out[i];
    }
    return;
  }
};

template <int BLOCK, int UNROLL, typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_reference(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                     const U32Div div) {
  KernelImpl().template Run<BLOCK, UNROLL, DivideRef>(cycles, out, dividends,
                                                      div);
}

template <int BLOCK, int UNROLL, typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
               const U32Div div) {
  KernelImpl().template Run<BLOCK, UNROLL, Divide>(cycles, out, dividends, div);
}

template <int BLOCK, int UNROLL, typename KernelImpl>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div_bounded(clock_t *cycles, uint32_t *out,
                       const uint32_t *dividends, const U32Div div) {
  KernelImpl().template Run<BLOCK, UNROLL, DivideBounded>(cycles, out,
                                                          dividends, div);
}

} // namespace impl

template <int BLOCK, int UNROLL>
struct KernelReference {
  void operator()(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_reference<BLOCK, UNROLL, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int BLOCK, int UNROLL>
struct KernelDiv {
  void operator()(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_div<BLOCK, UNROLL, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <int BLOCK, int UNROLL>
struct KernelDivBounded {
  void operator()(clock_t *cycles, uint32_t *out, const uint32_t *dividends,
                  const U32Div &div, int n_blocks, cudaStream_t stream) const {
    impl::kernel_div_bounded<BLOCK, UNROLL, impl::KernelImpl>
        <<<n_blocks, BLOCK, 0, stream>>>(cycles, out, dividends, div);
  }
};

template <template <int BLOCK, int UNROLL> typename Kernel>
class Test {

  static constexpr int UNROLL = 4;
  static constexpr int ELEM_PER_THREAD = TEST_COUNT / HOST_THREAD_COUNT;
  // we use only 1 block for benchmark
  static constexpr int CTA_COUNT = 1;

public:
  explicit Test(uint32_t d_, bool large_n_ = false)
      : thds(HOST_THREAD_COUNT), n_h(TEST_COUNT), out_h(TEST_COUNT),
        ref_h(TEST_COUNT), target_h(TEST_COUNT), div(d_), total_time_slow(0),
        total_time_fast(0), large_n(large_n_) {
    setup();
  }
  virtual ~Test() { cleanup(); }

  void Run() {
    prelude();

    /* run reference */
    if (!run_it(total_time_slow, cycles_slow, ref_h, ref_d, "reference",
                KernelReference<CTA_SIZE, UNROLL>())) {
      printf("[Error] reference kernel wrong answer\n");
      return;
    }

    /* run target */
    if (!run_it(total_time_fast, cycles_fast, target_h, target_d, "target",
                Kernel<CTA_SIZE, UNROLL>())) {
      return;
    }

    printf("d: %u,\treference: %lld us,\ttarget: %lld us\n", div.GetD(),
           static_cast<long long>(total_time_slow),
           static_cast<long long>(total_time_fast));
    return;
  }

private:
  template <typename Func>
  void host_threading(Func &&func) {
    if (thds.size() < HOST_THREAD_COUNT) {
      thds.resize(HOST_THREAD_COUNT);
    }
    for (int i = 0; i < HOST_THREAD_COUNT; ++i) {
      thds[i] = std::thread(std::forward<Func>(func), i);
    }
    for (int i = 0; i < HOST_THREAD_COUNT; ++i) {
      thds[i].join();
    }
    return;
  }

  void setup() {
    CHECK_CUDA(cudaMalloc(&n_d, n_h.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&ref_d, ref_h.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&target_d, target_h.size() * sizeof(uint32_t)));
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

  void prelude() {
    /* generate dividends */
    host_threading([&](int i) {
      using Dist = std::uniform_int_distribution<uint32_t>;
      std::default_random_engine rng(i);
      std::shared_ptr<Dist> dist;
      if (large_n) {
        dist = std::make_shared<Dist>(0, UINT32_MAX);
      } else {
        dist = std::make_shared<Dist>(0, INT32_MAX);
      }

      for (int j = 0; j < ELEM_PER_THREAD; ++j) {
        n_h[i * ELEM_PER_THREAD + j] = (*dist)(rng);
      }
    });

    /* copy to device */
    CHECK_CUDA(cudaMemcpyAsync(n_d, n_h.data(), n_h.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    /* run host */
    host_threading([&](int i) {
      for (int j = 0; j < ELEM_PER_THREAD; ++j) {
        out_h[i * ELEM_PER_THREAD + j] =
            n_h[i * ELEM_PER_THREAD + j] / div.GetD();
      }
    });
    return;
  }

  template <typename Func>
  bool run_it(clock_t &duration, clock_t *cycles,
              std::vector<uint32_t> &data_host, uint32_t *data_device,
              const char *name, Func &&func) {
    // warmup
    func(cycles, data_device, n_d, div, CTA_COUNT, stream);
    CHECK_KERNEL();
    // run
    func(cycles, data_device, n_d, div, CTA_COUNT, stream);
    CHECK_KERNEL();
    CHECK_CUDA(cudaMemcpyAsync(&duration, cycles, sizeof(clock_t),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(data_host.data(), data_device,
                               data_host.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // check
    std::vector<bool> passed(HOST_THREAD_COUNT, true);
    std::vector<std::string> errors(HOST_THREAD_COUNT);
    host_threading([&](int i) {
      for (int j = 0; j < ELEM_PER_THREAD; ++j) {
        int idx = i * ELEM_PER_THREAD + j;
        if (out_h[idx] != data_host[idx]) {
          char buf[512];
          snprintf(buf, sizeof(buf), "Error: %u / %u = %u, %s returns: %u",
                   n_h[idx], div.GetD(), out_h[idx], name, data_host[idx]);
          errors[i] = buf;
          passed[i] = false;
          break;
        }
      }
    });
    if (std::any_of(passed.begin(), passed.end(), [](bool x) { return !x; })) {
      for (int i = 0; i < HOST_THREAD_COUNT; ++i) {
        if (!passed[i]) {
          printf("%s\n", errors[i].c_str());
          return false;
        }
      }
    }
    return true;
  }

  uint32_t *n_d;
  uint32_t *ref_d;
  uint32_t *target_d;

  clock_t *cycles_slow;
  clock_t *cycles_fast;

  cudaStream_t stream;

  std::vector<std::thread> thds;
  std::vector<uint32_t> n_h;
  std::vector<uint32_t> out_h;
  std::vector<uint32_t> ref_h;
  std::vector<uint32_t> target_h;

  U32Div div;
  clock_t total_time_slow;
  clock_t total_time_fast;
  bool large_n;
};

using TestDivBounded = Test<KernelDivBounded>;
using TestDiv = Test<KernelDiv>;

int main() {
  constexpr int N_CASES = 5;
  puts("This is a test for correctness, NOT a benchmark.\n");

  srand((unsigned)time(nullptr));

  puts("DivBounded, d = rand() + 1");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = rand() + 1U;
    TestDivBounded test(d);
    test.Run();
  }

  puts("\nDivBounded, d = 2^31");
  {
    uint32_t d = (1U << 31);
    TestDivBounded test(d);
    test.Run();
  }

  puts("\nDivBounded, d > 2^31");
  {
    uint32_t d = UINT32_MAX - rand();
    TestDivBounded test(d);
    test.Run();
  }

  puts("\nThis is highly probable to fail for DivBounded due to n >= 2^31");
  {
    uint32_t d = rand() + 1U;
    TestDivBounded test(d, true);
    test.Run();
  }

  puts("\nDiv, d = UINT32_MAX - rand()");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = UINT32_MAX - rand();
    TestDiv test(d);
    test.Run();
  }

  puts("\nDiv, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()");
  for (int i = 0; i < N_CASES; i++) {
    uint32_t d = UINT32_MAX - rand();
    TestDiv test(d, true);
    test.Run();
  }
  return 0;
}
