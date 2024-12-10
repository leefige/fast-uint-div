#include "u32div.hpp"

#include <algorithm>
#include <chrono>
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

static constexpr int TEST_COUNT = 1 << 24;
static constexpr int HOST_THREAD_COUNT = 16;
static constexpr int ELEM_PER_THREAD = TEST_COUNT / HOST_THREAD_COUNT;
static_assert(TEST_COUNT % HOST_THREAD_COUNT == 0,
              "requires TEST_COUNT % HOST_THREAD_COUNT == 0");

namespace impl {

template <typename Func>
void threading(std::vector<std::thread> &thds, Func &&func) {
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

bool check(std::vector<std::thread> &thds, const std::vector<uint32_t> &ref,
           const std::vector<uint32_t> &target,
           const std::vector<uint32_t> &dividends, uint32_t d,
           const char *name) {
  std::vector<bool> passed(HOST_THREAD_COUNT, true);
  std::vector<std::string> errors(HOST_THREAD_COUNT);
  threading(thds, [&](int i) {
    for (int j = 0; j < ELEM_PER_THREAD; ++j) {
      int idx = i * ELEM_PER_THREAD + j;
      if (ref[idx] != target[idx]) {
        char buf[512];
        snprintf(buf, sizeof(buf), "Error: %u / %u = %u, %s returns: %u",
                 dividends[idx], d, ref[idx], name, target[idx]);
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

template <typename Func>
void test_impl(const U32Div &div, bool large_n, const char *name, Func &&func) {
  uint32_t d = div.GetD();
  long long total_time_slow = 0;
  long long total_time_fast = 0;

  std::vector<uint32_t> dividends(TEST_COUNT);
  std::vector<uint32_t> out_slow(TEST_COUNT);
  std::vector<uint32_t> out_fast(TEST_COUNT);

  std::vector<std::thread> thds(HOST_THREAD_COUNT);
  threading(thds, [&](int i) {
    using Dist = std::uniform_int_distribution<uint32_t>;
    std::default_random_engine rng(i);
    std::shared_ptr<Dist> dist;
    if (large_n) {
      dist = std::make_shared<Dist>(0, UINT32_MAX);
    } else {
      dist = std::make_shared<Dist>(0, INT32_MAX);
    }

    for (int j = 0; j < ELEM_PER_THREAD; ++j) {
      dividends[i * ELEM_PER_THREAD + j] = (*dist)(rng);
    }
  });

  // slow
  auto start_slow = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TEST_COUNT; ++i) {
    out_slow[i] = dividends[i] / d;
  }
  auto end_slow = std::chrono::high_resolution_clock::now();
  auto duration_slow = std::chrono::duration_cast<std::chrono::microseconds>(
      end_slow - start_slow);
  total_time_slow = duration_slow.count();

  // fast
  auto start_fast = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < TEST_COUNT; ++i) {
    out_fast[i] = func(dividends[i], div);
  }
  auto end_fast = std::chrono::high_resolution_clock::now();
  auto duration_fast = std::chrono::duration_cast<std::chrono::microseconds>(
      end_fast - start_fast);
  total_time_fast = duration_fast.count();

  if (!impl::check(thds, out_slow, out_fast, dividends, d, name)) {
    return;
  }

  printf("d: %u,\tslow: %lld us,\tfast: %lld us,\tspeedup: %lf\n", d,
         total_time_slow, total_time_fast,
         (double)total_time_slow / (double)total_time_fast);
  return;
}

} // namespace impl

void test_div_bounded(const U32Div &div, bool large_n = false) {
  impl::test_impl(
      div, large_n, "DivBounded",
      [&](uint32_t n, const U32Div &div) { return div.DivBounded(n); });
  return;
}

void test_div(const U32Div &div, bool large_n = false) {
  impl::test_impl(div, large_n, "Div",
                  [&](uint32_t n, const U32Div &div) { return div.Div(n); });
  return;
}

int main() {
  srand((unsigned)time(nullptr));

  puts("DivBounded, d = rand() + 1");
  for (int i = 0; i < 10; i++) {
    uint32_t d = rand() + 1U;

    U32Div div(d);
    test_div_bounded(div);
  }

  puts("\nDivBounded, d = 2^31");
  {
    uint32_t d = (1U << 31);
    U32Div div(d);
    test_div_bounded(div);
  }

  puts("\nThis is intended to fail for DivBounded due to d > 2^31");
  {
    uint32_t d = (1U << 31) + 1U;
    U32Div div(d);
    test_div_bounded(div);
  }

  puts("\nThis is highly probable to fail for DivBounded due to n >= 2^31");
  {
    uint32_t d = rand() + 1U;
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
