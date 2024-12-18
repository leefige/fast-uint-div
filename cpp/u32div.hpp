/*!
 * Copyright (c) 2024 Yifei Li
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

/**
 * @brief Fast unsigned integer division by a constant.
 *
 * The divisor must be a positive integer less than 2^32.
 *
 * @throws std::invalid_argument if d == 0.
 */
class alignas(sizeof(uint32_t)) U32Div {
  static constexpr int BIT_WIDTH = 8 * sizeof(uint32_t);

public:
  explicit U32Div(uint32_t d) : _d(d), _shift(log2up(d)) {
    if (d == 0) {
      throw std::invalid_argument("Divisor must be non-zero");
    }

    uint32_t k = BIT_WIDTH + _shift;
    // if k >= 2 * BIT_WIDTH, (uint64_t(1) << k) will overflow
    uint64_t m =
        _shift < BIT_WIDTH
            ? ((uint64_t(1) << k) + (d - 1) - (((uint64_t(1) << k) - 1) % d)) /
                  d
            : static_cast<uint64_t>(ceil(pow(2, k) / d));
    _magic = static_cast<uint32_t>(m);

    // handles d == 1
    _sh_0 = _shift < 1U ? 0U : 1U;
    _sh_1 = _shift - _sh_0;
  }

  /**
   * @brief Get the divisor.
   *
   * @return uint32_t
   */
  uint32_t GetD() const { return _d; }

  /**
   * @brief Compute n / d, for bounded n (and d for some architectures).
   *
   * Requires n < 2^31.
   *
   * May require d <= 2^31, depending on architecture, for example:
   *  (1) for x86 or x86_64, d <= 2^31;
   *  (2) for Arm, d can be any positive 32-bit integer.
   *
   * @note The archtecture-dependent limitation on d is due to the definition of
   * logical shift right intruction for (N-bit integer >> N). For x86, it is
   * undefined. While for Arm, it is well-defined.
   *
   * @param n Dividend, n < 2^31.
   * @return uint32_t
   */
  uint32_t DivBounded(uint32_t n) const {
    return (umulhi(n, _magic) + n) >> _shift;
  }

  /**
   * @brief Compute n / d, for all possible values of n and d.
   *
   * @param n Dividend.
   * @return uint32_t
   */
  uint32_t Div(uint32_t n) const {
    uint32_t q = umulhi(n, _magic);
    return (((n - q) >> _sh_0) + q) >> _sh_1;
  }

private:
  /**
   * @brief Compute the high half of a * b.
   *
   * @param a
   * @param b
   * @return uint32_t
   */
  static inline uint32_t umulhi(uint32_t a, uint32_t b) {
#if defined(__x86_64__) || defined(__i386__)
    // x86_84 BMI2 expansion provides mulx for faster umulhi.
    // GCC and clang only.
    uint32_t hi;
    __asm__("mulxl %2, %%eax, %%eax;" : "=a"(hi) : "d"(b), "rm"(a));
    return hi;
#else
    return ((uint64_t)a * b) >> 32;
#endif
  }

  /**
   * @brief Compute the round-down integer log2 of x.
   *
   * @note If x == 0, the result is undefined.
   *
   * @param x Non-zero unsigned integer.
   * @return uint32_t
   */
  static inline uint32_t log2down(uint32_t x) {
#if defined(_MSC_VER)
    return 31U - __lzcnt(x);
#else
    return 31U - __builtin_clz(x);
#endif
  }

  /**
   * @brief Compute the round-up integer log2 of x.
   *
   * @note If x == 0, the result is undefined.
   *
   * @param x Non-zero unsigned integer.
   * @return uint32_t
   */
  static inline uint32_t log2up(uint32_t x) {
    return x <= 1 ? 0 : log2down(x - 1) + 1;
  }

  uint32_t _d;
  uint32_t _shift;
  uint32_t _magic;
  uint32_t _sh_0;
  uint32_t _sh_1;
};
