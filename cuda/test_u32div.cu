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
static constexpr int CTA_SIZE = 256;
static constexpr int TEST_COUNT = 1 << 24;
static_assert(TEST_COUNT % HOST_THREAD_COUNT == 0,
              "requires TEST_COUNT % HOST_THREAD_COUNT == 0");

namespace impl {

template <int BLOCK, int UNROLL, typename Func>
__device__ __forceinline__ void kernel_impl(uint32_t *out,
                                            const uint32_t *dividends,
                                            const U32Div &div, Func &&func) {
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
      v_out[i].data[k] = func(v_in[i].data[k], div);
    }
  }

  VecT *out_ptr =
      reinterpret_cast<VecT *>(out) + blockIdx.x * BLOCK * UNROLL + threadIdx.x;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    *(out_ptr + i * BLOCK) = v_out[i];
  }
  return;
}

} // namespace impl

template <int BLOCK, int UNROLL>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_reference(uint32_t *out, const uint32_t *dividends,
                     const U32Div div) {
  impl::kernel_impl<BLOCK, UNROLL>(out, dividends, div, impl::DivideRef());
}

template <int BLOCK, int UNROLL>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div(uint32_t *out, const uint32_t *dividends, const U32Div div) {
  impl::kernel_impl<BLOCK, UNROLL>(out, dividends, div, impl::Divide());
}

template <int BLOCK, int UNROLL>
__global__ void __launch_bounds__(BLOCK, 1)
    kernel_div_bounded(uint32_t *out, const uint32_t *dividends,
                       const U32Div div) {
  impl::kernel_impl<BLOCK, UNROLL>(out, dividends, div, impl::DivideBounded());
}

class TestBase {

protected:
  virtual void launch_kernel(uint32_t *out, const uint32_t *dividends,
                             const U32Div &div, cudaStream_t stream) = 0;

  static constexpr int UNROLL = 4;
  static constexpr int ELEM_PER_THREAD = TEST_COUNT / HOST_THREAD_COUNT;
  static constexpr int CTA_COUNT =
      TEST_COUNT / (CTA_SIZE * UNROLL * 16 / sizeof(uint32_t));

public:
  explicit TestBase(uint32_t d_, bool large_n_)
      : thds(HOST_THREAD_COUNT), n_h(TEST_COUNT), out_h(TEST_COUNT),
        ref_h(TEST_COUNT), target_h(TEST_COUNT), div(d_), total_time_slow(0),
        total_time_fast(0), large_n(large_n_) {
    setup();
  }
  virtual ~TestBase() { cleanup(); }

  void Run() {
    prelude();

    /* run reference */
    if (!time_it(total_time_slow, ref_start, ref_stop, ref_h, ref_d,
                 "reference", [&] {
                   kernel_reference<CTA_SIZE, UNROLL>
                       <<<CTA_COUNT, CTA_SIZE, 0, stream>>>(ref_d, n_d, div);
                 })) {
      printf("[Error] reference kernel wrong answer\n");
      return;
    }

    /* run target */
    if (!time_it(total_time_fast, target_start, target_stop, target_h, target_d,
                 "target",
                 [&] { launch_kernel(target_d, n_d, div, stream); })) {
      return;
    }

    total_time_slow *= 1000;
    total_time_fast *= 1000;
    printf("d: %u,\treference: %.2f us,\ttarget: %.2f us\n", div.GetD(),
           total_time_slow, total_time_fast);
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
    CHECK_CUDA(cudaEventCreate(&ref_start));
    CHECK_CUDA(cudaEventCreate(&ref_stop));
    CHECK_CUDA(cudaEventCreate(&target_start));
    CHECK_CUDA(cudaEventCreate(&target_stop));
    CHECK_CUDA(cudaStreamCreate(&stream));
    return;
  }

  void cleanup() {
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(target_start));
    CHECK_CUDA(cudaEventDestroy(target_stop));
    CHECK_CUDA(cudaEventDestroy(ref_start));
    CHECK_CUDA(cudaEventDestroy(ref_stop));
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
  bool time_it(float &duration, cudaEvent_t start, cudaEvent_t stop,
               std::vector<uint32_t> &data_host, uint32_t *data_device,
               const char *name, Func &&func) {
    // warmup
    func();
    CHECK_KERNEL();
    // run
    CHECK_CUDA(cudaEventRecord(start, stream));
    func();
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&duration, start, stop));
    CHECK_CUDA(cudaMemcpy(data_host.data(), data_device,
                          data_host.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
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

  cudaEvent_t ref_start;
  cudaEvent_t ref_stop;
  cudaEvent_t target_start;
  cudaEvent_t target_stop;
  cudaStream_t stream;

  std::vector<std::thread> thds;
  std::vector<uint32_t> n_h;
  std::vector<uint32_t> out_h;
  std::vector<uint32_t> ref_h;
  std::vector<uint32_t> target_h;

  U32Div div;
  float total_time_slow;
  float total_time_fast;
  bool large_n;
};

class TestDiv : public TestBase {
public:
  explicit TestDiv(uint32_t d_, bool large_n_ = false)
      : TestBase(d_, large_n_) {}

protected:
  virtual void launch_kernel(uint32_t *out, const uint32_t *dividends,
                             const U32Div &div, cudaStream_t stream) override {
    kernel_div<CTA_SIZE, UNROLL>
        <<<CTA_COUNT, CTA_SIZE, 0, stream>>>(out, dividends, div);
    return;
  }
};

class TestDivBounded : public TestBase {
public:
  explicit TestDivBounded(uint32_t d_, bool large_n_ = false)
      : TestBase(d_, large_n_) {}

protected:
  virtual void launch_kernel(uint32_t *out, const uint32_t *dividends,
                             const U32Div &div, cudaStream_t stream) override {
    kernel_div_bounded<CTA_SIZE, UNROLL>
        <<<CTA_COUNT, CTA_SIZE, 0, stream>>>(out, dividends, div);
    return;
  }
};

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
