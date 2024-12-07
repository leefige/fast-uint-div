#include "u32div.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define TEST_COUNT 1000000

void test_div_bounded(const U32Div &div, bool large_n = false) {
  uint32_t d = div.GetD();
  int64_t total_time_slow = 0;
  int64_t total_time_fast = 0;

  for (int i = 0; i < TEST_COUNT; ++i) {
    uint32_t n = large_n ? UINT32_MAX - rand() : rand();

    auto start_slow = std::chrono::high_resolution_clock::now();
    uint32_t q_slow = n / d;
    auto end_slow = std::chrono::high_resolution_clock::now();

    auto start_fast = std::chrono::high_resolution_clock::now();
    uint32_t q_fast = div.DivBounded(n);
    auto end_fast = std::chrono::high_resolution_clock::now();

    if (q_slow != q_fast) {
      printf("Error: %u / %u = %u, DivBounded returns: %u\n", n, d, q_slow,
             q_fast);
      return;
    }

    auto duration_slow = std::chrono::duration_cast<std::chrono::microseconds>(
        end_slow - start_slow);
    auto duration_fast = std::chrono::duration_cast<std::chrono::microseconds>(
        end_fast - start_fast);
    total_time_slow += duration_slow.count();
    total_time_fast += duration_fast.count();
  }

  printf("d: %u,\tslow: %lld us,\tfast: %lld us,\tspeedup: %f\n", d,
         total_time_slow, total_time_fast,
         (float)total_time_slow / total_time_fast);
  return;
}

void test_div(const U32Div &div, bool large_n = false) {
  uint32_t d = div.GetD();
  int64_t total_time_slow = 0;
  int64_t total_time_fast = 0;

  for (int i = 0; i < TEST_COUNT; ++i) {
    uint32_t n = large_n ? UINT32_MAX - rand() : rand();

    auto start_slow = std::chrono::high_resolution_clock::now();
    uint32_t q_slow = n / d;
    auto end_slow = std::chrono::high_resolution_clock::now();

    auto start_fast = std::chrono::high_resolution_clock::now();
    uint32_t q_fast = div.Div(n);
    auto end_fast = std::chrono::high_resolution_clock::now();

    if (q_slow != q_fast) {
      printf("Error: %u / %u = %u, Div returns: %u\n", n, d, q_slow, q_fast);
      return;
    }

    auto duration_slow = std::chrono::duration_cast<std::chrono::microseconds>(
        end_slow - start_slow);
    auto duration_fast = std::chrono::duration_cast<std::chrono::microseconds>(
        end_fast - start_fast);
    total_time_slow += duration_slow.count();
    total_time_fast += duration_fast.count();
  }

  printf("d: %u,\tslow: %lld us,\tfast: %lld us,\tspeedup: %f\n", d,
         total_time_slow, total_time_fast,
         (float)total_time_slow / total_time_fast);
  return;
}

int main() {
  srand(time(nullptr));

  puts("DivBounded, d = rand() + 1");
  for (int i = 0; i < 10; i++) {
    uint32_t d = rand() + 1;

    U32Div div(d);
    test_div_bounded(div);
  }

  puts("\nDivBounded, d = 2^31");
  {
    uint32_t d = (1 << 31);
    U32Div div(d);
    test_div_bounded(div);
  }

  puts("\nThis is intended to fail for DivBounded due to d > 2^31");
  {
    uint32_t d = (1 << 31) + 1;
    U32Div div(d);
    test_div_bounded(div);
  }

  puts("\nThis is highly probable to fail for DivBounded due to n >= 2^31");
  {
    uint32_t d = rand() + 1;
    U32Div div(d);
    test_div_bounded(div, true);
  }

  puts("\nDiv, d = UINT32_MAX - rand()");
  for (int i = 0; i < 10; i++) {
    uint32_t d = UINT32_MAX - rand();
    U32Div div(d);
    test_div(div);
  }

  puts("\nDiv, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()");
  for (int i = 0; i < 10; i++) {
    uint32_t d = UINT32_MAX - rand();
    U32Div div(d);
    test_div(div, true);
  }

  return 0;
}
