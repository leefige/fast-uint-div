#include "u32div.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <thread>
#include <vector>

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

static constexpr int WARP_SIZE = 32;
static constexpr int TEST_COUNT = 1 << 23;
static constexpr int THREAD_COUNT = 16;

template <int N, typename T> struct alignas(sizeof(uint4)) Vec {
  T data[N];
};

// one thread computes all elements
__global__ void reference(uint32_t *out, const uint32_t *dividends,
                          const U32Div div) {
  const uint32_t d = div.GetD();

  constexpr int UNROLL = 16 / sizeof(uint32_t);
  static_assert(TEST_COUNT % UNROLL == 0, "requires TEST_COUNT % UNROLL == 0");
  using VecT = Vec<UNROLL, uint32_t>;
  static_assert(sizeof(VecT) == sizeof(uint4),
                "assert sizeof(VecT) == sizeof(uint4)");

  constexpr int N_LOOP = TEST_COUNT / UNROLL;
  for (int i = 0; i < N_LOOP; ++i) {
    VecT v_in;
    const VecT *in_ptr = reinterpret_cast<const VecT *>(dividends) + i;
    v_in = *in_ptr;

    VecT v_out;
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
      v_out.data[k] = v_in.data[k] / d;
    }

    VecT *out_ptr = reinterpret_cast<VecT *>(out) + i;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *out_ptr = v_out;
    }
  }
  return;
}

void test_body(const U32Div &div, bool large_n = false) {
  uint32_t d = div.GetD();

  float total_time_slow = 1;
  float total_time_fast = 1;

  std::vector<uint32_t> n_h(TEST_COUNT);
  std::vector<std::thread> thds(THREAD_COUNT);

  /* Step 1: generate dividends */
  static_assert(TEST_COUNT % THREAD_COUNT == 0,
                "requires TEST_COUNT % THREAD_COUNT == 0");
  constexpr int ELEM_PER_THREAD = TEST_COUNT / THREAD_COUNT;
  for (int i = 0; i < THREAD_COUNT; ++i) {
    thds[i] = std::thread([&, i]() {
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
  }
  for (int i = 0; i < THREAD_COUNT; ++i) {
    thds[i].join();
  }

  /* Step 2: run host */
  std::vector<uint32_t> out_h(TEST_COUNT);
  for (int i = 0; i < THREAD_COUNT; ++i) {
    thds[i] = std::thread([&, i]() {
      for (int j = 0; j < ELEM_PER_THREAD; ++j) {
        out_h[i * ELEM_PER_THREAD + j] = n_h[i * ELEM_PER_THREAD + j] / d;
      }
    });
  }
  for (int i = 0; i < THREAD_COUNT; ++i) {
    thds[i].join();
  }

  /* Step 3: copy dividends to device */
  uint32_t *n_d;
  CHECK_CUDA(cudaMalloc(&n_d, n_h.size() * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(n_d, n_h.data(), n_h.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  /* Step 4: run reference */
  std::vector<uint32_t> ref_h(TEST_COUNT);
  uint32_t *ref_d;
  CHECK_CUDA(cudaMalloc(&ref_d, ref_h.size() * sizeof(uint32_t)));
  cudaEvent_t ref_start, ref_stop;
  CHECK_CUDA(cudaEventCreate(&ref_start));
  CHECK_CUDA(cudaEventCreate(&ref_stop));
  // warmup
  reference<<<1, WARP_SIZE>>>(ref_d, n_d, div);
  CHECK_KERNEL();
  // run
  CHECK_CUDA(cudaEventRecord(ref_start));
  reference<<<1, WARP_SIZE>>>(ref_d, n_d, div);
  CHECK_KERNEL();
  CHECK_CUDA(cudaEventRecord(ref_stop));
  CHECK_CUDA(cudaEventSynchronize(ref_stop));
  CHECK_CUDA(cudaEventElapsedTime(&total_time_slow, ref_start, ref_stop));
  CHECK_CUDA(cudaEventDestroy(ref_start));
  CHECK_CUDA(cudaEventDestroy(ref_stop));
  // check
  CHECK_CUDA(cudaMemcpy(ref_h.data(), ref_d, ref_h.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(ref_d));
  for (int i = 0; i < THREAD_COUNT; ++i) {
    thds[i] = std::thread([&, i]() {
      for (int j = 0; j < ELEM_PER_THREAD; ++j) {
        int idx = i * ELEM_PER_THREAD + j;
        if (out_h[idx] != ref_h[idx]) {
          printf("Error: %u / %u = %u, Ref returns: %u\n", n_h[idx], d,
                 out_h[idx], ref_h[idx]);
          break;
        }
      }
    });
  }
  for (int i = 0; i < THREAD_COUNT; ++i) {
    thds[i].join();
  }

  // free
  CHECK_CUDA(cudaFree(n_d));

  total_time_slow *= 1000;
  total_time_fast *= 1000;
  printf("d: %u,\tslow: %f us,\tfast: %f us,\tspeedup: %f\n", d,
         total_time_slow, total_time_fast, total_time_slow / total_time_fast);
  return;
}

int main() {
  srand((unsigned)time(nullptr));

  puts("DivBounded, d = rand() + 1");
  for (int i = 0; i < 5; i++) {
    uint32_t d = rand() + 1U;

    U32Div div(d);
    test_body(div);
  }

  return 0;
}
