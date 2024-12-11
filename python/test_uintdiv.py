# Copyright (c) 2024 Yifei Li
# SPDX-License-Identifier: MIT

from uintdiv import U32Div
from random import randint, seed
import time
from typing import Callable

TEST_COUNT = 1 << 14
N_BIT = 32


def test_impl(div: U32Div, large_n: bool, func: Callable):
    d = div.d
    dividends = [
        randint(0, 2**N_BIT - 1 if large_n else 2 ** (N_BIT - 1) - 1)
        for _ in range(TEST_COUNT)
    ]
    out_ref = [0 for _ in range(TEST_COUNT)]
    out_target = [0 for _ in range(TEST_COUNT)]

    start_ref = time.perf_counter_ns()
    for i in range(TEST_COUNT):
        out_ref[i] = dividends[i] // d
    end_ref = time.perf_counter_ns()

    start_target = time.perf_counter_ns()
    for i in range(TEST_COUNT):
        out_target[i] = func(dividends[i], div)
    end_target = time.perf_counter_ns()

    for i in range(TEST_COUNT):
        if out_ref[i] != out_target[i]:
            print(
                f"Error: {dividends[i]} / {d} = {out_ref[i]}, div returns {out_target[i]}"
            )
            return

    total_time_ref = (end_ref - start_ref) / 1e3
    total_time_target = (end_target - start_target) / 1e3
    print(
        f"d: {d},\treference: {int(total_time_ref)} us,\ttarget: {int(total_time_target)} us"
    )


def test_div(div: U32Div, large_n: bool = False):
    test_impl(div, large_n, lambda n, div: div.div(n))


def main():
    seed(time.time())

    print("d <= 2**31, n < 2**31")
    for _ in range(3):
        d = randint(1, 2 ** (N_BIT - 1))
        div = U32Div(d)
        test_div(div)

    print("\nd = 2**31, n < 2**31")
    d = 2 ** (N_BIT - 1)
    div = U32Div(d)
    test_div(div)

    print("\nd = 2**31 + 1, n < 2**31")
    d = 2 ** (N_BIT - 1) + 1
    div = U32Div(d)
    test_div(div)

    print("\nd <= 2**31, n >= 2**31")
    for _ in range(3):
        d = randint(1, 2 ** (N_BIT - 1))
        div = U32Div(d)
        test_div(div, large_n=True)

    print("\nd >= 2**31, n < 2**31")
    for _ in range(3):
        d = randint(2 ** (N_BIT - 1), 2**N_BIT - 1)
        div = U32Div(d)
        test_div(div)

    print("\nd >= 2**31, n >= 2**31")
    for _ in range(3):
        d = randint(2 ** (N_BIT - 1), 2**N_BIT - 1)
        div = U32Div(d)
        test_div(div, large_n=True)


if __name__ == "__main__":
    main()
