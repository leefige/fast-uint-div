# Copyright (c) 2024 Yifei Li
# SPDX-License-Identifier: MIT

import math


class UintDiv(object):

    def __init__(self, d: int, Nbit: int):
        """Unsigned integer division by a constant.

        This is a proof of concept, and the performance is not good.

        Args:
            d (int): non-negative divisor d < pow(2, Nbit)
            Nbit (int): number of bits of the dividend and the divisor
        """
        assert Nbit > 0
        assert d > 0
        assert d < (1 << Nbit)
        self.__d = d
        self.__Nbit = Nbit
        self.__shift = math.ceil(math.log2(d))

        k = Nbit + self.__shift
        m = math.ceil(math.pow(2, k) / d)
        self.__magic = m & ((1 << Nbit) - 1)

    @property
    def d(self) -> int:
        """Get the divisor.

        Returns:
            int: divisor
        """
        return self.__d

    def div(self, n: int) -> int:
        """Compute n // d, for n in [0, pow(2, Nbit)).

        Args:
            n (int): non-negative dividend n < pow(2, Nbit)

        Returns:
            int: value of n // d

        Note:
            This will NOT overflow for n >= pow(2, Nbit - 1),
            because Python supports arbitrary-precision integers.
        """
        return (self.__umulhi(n, self.__magic) + n) >> self.__shift

    def __umulhi(self, a: int, b: int) -> int:
        """Compute high Nbit of a * b.

        Args:
            a (int): non-negative a < pow(2, Nbit)
            b (int): non-negative b < pow(2, Nbit)

        Returns:
            int: high Nbit of a * b
        """
        return (a * b) >> self.__Nbit


class U32Div(UintDiv):

    def __init__(self, d: int):
        """Fast unsigned 32-bit integer division by a constant.

        Args:
            d (int): non-negative divisor d < pow(2, 32)
        """
        super().__init__(d, 32)
