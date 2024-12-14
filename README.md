# Fast Unsigned Integer Divsion by Constants

## Build and run

### Requirements

- CMake version 3.24 or higher
- C++ compiler with C++17 support
- For CUDA: CUDA 12.0 or higher
- For Python: Python 3.10 or higher

### C++

Unix-like systems:

Build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -v
```

Run:

```sh
build/cpp/uint-div-cpp
```

Windows with PowerShell:

Build:

```powershell
cmake -S . -B build
cmake --build build --config Release -v
```

Run:

```powershell
.\build\cpp\Release\uint-div-cpp.exe
```

### CUDA

Unix-like systems:

Build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON
cmake --build build -v
```

Run:

```sh
build/cuda/uint-div-cuda
```

Windows with PowerShell:

Build:

```powershell
cmake -S . -B build -DBUILD_CUDA=ON
cmake --build build --config Release -v
```

Run:

```powershell
.\build\cuda\Release\uint-div-cuda.exe
```

## Experiments

### C++ benchmark

A single thread sequentially computes 2^24 (1 << 24) unsigned integer divisions.
The baseline ("slow") is the built-in integer division (`operator/`), which generally maps to the architecture-provided integer division instruction.
The fast division ("fast") is implemented by either `DivBounded` or `Div`.

Some conclusions:

1. Intel Core i7 (x86_64) with MSVC: `DivBounded` achieves up to 2.4x speedup, `Div` achieves up to 1.9x speedup.
2. Intel Core i7 (x86_64) with GCC: `DivBounded` achieves up to 2.7x speedup, `Div` achieves up to 1.9x speedup.
3. Intel Xeon (x86_64): `DivBounded` achieves about 2x speedup, `Div` achieves 1.6~1.8x speedup.
4. Apple M3 (AArch64): `DivBounded` achieves up to 6x speedup, `Div` achieves up to 4.4x speedup.
5. x86 with GCC performs slightly better than with MSVC.
6. Fast division far outperforms the built-in integer division on AArch64, and shows obvious speedup on all platforms.
7. `DivBounded` is generally faster than `Div`.
8. `DivBounded` does not support d > 2^31 on x86, while it does support d > 2^31 on AArch64

#### Intel Core

CPU: Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz

OS: Windows 11 23H2

Compiler: MSVC 19.41.34123

```plain
DivBounded, d = rand() + 1
d: 12039,       slow: 23959 us, fast: 9875 us,  speedup: 2.426228
d: 27500,       slow: 23400 us, fast: 9761 us,  speedup: 2.397295
d: 17315,       slow: 22298 us, fast: 9882 us,  speedup: 2.256426
d: 31677,       slow: 23637 us, fast: 10058 us, speedup: 2.350070
d: 5728,        slow: 22172 us, fast: 11002 us, speedup: 2.015270
d: 11794,       slow: 22775 us, fast: 9958 us,  speedup: 2.287106
d: 24282,       slow: 22592 us, fast: 9915 us,  speedup: 2.278568
d: 6365,        slow: 22572 us, fast: 10035 us, speedup: 2.249327
d: 10737,       slow: 22663 us, fast: 9795 us,  speedup: 2.313731
d: 3276,        slow: 23110 us, fast: 9777 us,  speedup: 2.363711

DivBounded, d = 2^31
d: 2147483648,  slow: 22482 us, fast: 9770 us,  speedup: 2.301126

This is intended to fail for DivBounded due to d > 2^31
Error: 1178568022 / 2147483649 = 0, DivBounded returns: 2357136042

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3626093760 / 26536 = 136648, DivBounded returns: 5576

Div, d = UINT32_MAX - rand()
d: 4294952008,  slow: 23162 us, fast: 12518 us, speedup: 1.850296
d: 4294966064,  slow: 22596 us, fast: 12272 us, speedup: 1.841265
d: 4294950877,  slow: 22747 us, fast: 12173 us, speedup: 1.868644
d: 4294945016,  slow: 22472 us, fast: 12064 us, speedup: 1.862732
d: 4294946385,  slow: 22739 us, fast: 12435 us, speedup: 1.828629
d: 4294961689,  slow: 22489 us, fast: 12119 us, speedup: 1.855681
d: 4294961620,  slow: 22773 us, fast: 12280 us, speedup: 1.854479
d: 4294954426,  slow: 22535 us, fast: 12289 us, speedup: 1.833754
d: 4294951890,  slow: 22533 us, fast: 12202 us, speedup: 1.846664
d: 4294957822,  slow: 22522 us, fast: 11956 us, speedup: 1.883740

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 4294954159,  slow: 22974 us, fast: 12308 us, speedup: 1.866591
d: 4294944589,  slow: 22876 us, fast: 12263 us, speedup: 1.865449
d: 4294953074,  slow: 22613 us, fast: 12232 us, speedup: 1.848676
d: 4294955482,  slow: 22300 us, fast: 12036 us, speedup: 1.852775
d: 4294946733,  slow: 23015 us, fast: 12249 us, speedup: 1.878929
d: 4294944355,  slow: 22623 us, fast: 12325 us, speedup: 1.835538
d: 4294956742,  slow: 22585 us, fast: 12298 us, speedup: 1.836477
d: 4294965467,  slow: 22820 us, fast: 12620 us, speedup: 1.808241
d: 4294960752,  slow: 22626 us, fast: 12345 us, speedup: 1.832807
d: 4294941058,  slow: 22617 us, fast: 12211 us, speedup: 1.852182
```

CPU: Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz

OS: Ubuntu 22.04.3 LTS (WSL2)

Compiler: GCC 11.4.0

```plain
DivBounded, d = rand() + 1
d: 739410812,   slow: 22705 us, fast: 8305 us,  speedup: 2.733895
d: 1148822516,  slow: 21879 us, fast: 8814 us,  speedup: 2.482301
d: 1349944069,  slow: 21993 us, fast: 8426 us,  speedup: 2.610135
d: 983815658,   slow: 22587 us, fast: 8763 us,  speedup: 2.577542
d: 947712319,   slow: 22224 us, fast: 8213 us,  speedup: 2.705954
d: 914447778,   slow: 22781 us, fast: 8607 us,  speedup: 2.646799
d: 2036792398,  slow: 22560 us, fast: 9470 us,  speedup: 2.382260
d: 229551411,   slow: 22662 us, fast: 8367 us,  speedup: 2.708498
d: 2030602121,  slow: 22278 us, fast: 8568 us,  speedup: 2.600140
d: 1232154477,  slow: 22581 us, fast: 8305 us,  speedup: 2.718964

DivBounded, d = 2^31
d: 2147483648,  slow: 22077 us, fast: 8067 us,  speedup: 2.736705

This is intended to fail for DivBounded due to d > 2^31
Error: 282475248 / 2147483649 = 0, DivBounded returns: 564950495

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3622316814 / 1711391103 = 2, DivBounded returns: 0

Div, d = UINT32_MAX - rand()
d: 4261305535,  slow: 22344 us, fast: 11752 us, speedup: 1.901293
d: 2196503405,  slow: 21777 us, fast: 11786 us, speedup: 1.847701
d: 4273160857,  slow: 22074 us, fast: 11987 us, speedup: 1.841495
d: 4139866617,  slow: 22697 us, fast: 12163 us, speedup: 1.866069
d: 3608088022,  slow: 22061 us, fast: 12054 us, speedup: 1.830181
d: 2632662772,  slow: 22442 us, fast: 11935 us, speedup: 1.880352
d: 3156046287,  slow: 22560 us, fast: 12110 us, speedup: 1.862923
d: 2799603446,  slow: 22463 us, fast: 11823 us, speedup: 1.899941
d: 2431861227,  slow: 21872 us, fast: 11724 us, speedup: 1.865575
d: 4002122618,  slow: 22236 us, fast: 11895 us, speedup: 1.869357

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 4160742264,  slow: 22269 us, fast: 12024 us, speedup: 1.852046
d: 2620791515,  slow: 22276 us, fast: 11823 us, speedup: 1.884124
d: 4034141562,  slow: 22597 us, fast: 11996 us, speedup: 1.883711
d: 2724674569,  slow: 22660 us, fast: 12273 us, speedup: 1.846329
d: 3305279962,  slow: 22397 us, fast: 12094 us, speedup: 1.851910
d: 3005378867,  slow: 22103 us, fast: 11869 us, speedup: 1.862246
d: 2492167412,  slow: 22074 us, fast: 12029 us, speedup: 1.835065
d: 3567370893,  slow: 22287 us, fast: 12106 us, speedup: 1.840988
d: 3818778800,  slow: 22144 us, fast: 11973 us, speedup: 1.849495
d: 3430838036,  slow: 22359 us, fast: 12597 us, speedup: 1.774946
```

#### Intel Xeon

CPU: Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz

OS: Ubuntu 22.04.2 LTS

Compiler: GCC 11.3.0

```plain
DivBounded, d = rand() + 1
d: 1194173319,	slow: 45440 us,	fast: 19554 us,	speedup: 2.323821
d: 1466172372,	slow: 36744 us,	fast: 18802 us,	speedup: 1.954260
d: 326670292,	slow: 36020 us,	fast: 18347 us,	speedup: 1.963264
d: 134546121,	slow: 40543 us,	fast: 19003 us,	speedup: 2.133505
d: 1522997473,	slow: 30889 us,	fast: 16497 us,	speedup: 1.872401
d: 1919922272,	slow: 34882 us,	fast: 18204 us,	speedup: 1.916172
d: 236747362,	slow: 32104 us,	fast: 16051 us,	speedup: 2.000125
d: 1914636498,	slow: 34839 us,	fast: 18156 us,	speedup: 1.918870
d: 370689008,	slow: 35035 us,	fast: 18218 us,	speedup: 1.923098
d: 1943806296,	slow: 40208 us,	fast: 18622 us,	speedup: 2.159167

DivBounded, d = 2^31
d: 2147483648,	slow: 41588 us,	fast: 18973 us,	speedup: 2.191957

This is intended to fail for DivBounded due to d > 2^31
Error: 282475248 / 2147483649 = 0, DivBounded returns: 564950495

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3262921810 / 726827746 = 4, DivBounded returns: 0

Div, d = UINT32_MAX - rand()
d: 3054236855,	slow: 50627 us,	fast: 22552 us,	speedup: 2.244901
d: 3077386257,	slow: 36080 us,	fast: 22456 us,	speedup: 1.606698
d: 4133505628,	slow: 31541 us,	fast: 19154 us,	speedup: 1.646706
d: 2693242931,	slow: 32540 us,	fast: 19696 us,	speedup: 1.652112
d: 4169759927,	slow: 34131 us,	fast: 21212 us,	speedup: 1.609042
d: 2819937664,	slow: 34401 us,	fast: 22007 us,	speedup: 1.563184
d: 3542466851,	slow: 35155 us,	fast: 21695 us,	speedup: 1.620419
d: 4174289017,	slow: 35066 us,	fast: 21682 us,	speedup: 1.617286
d: 2642061525,	slow: 36099 us,	fast: 22321 us,	speedup: 1.617266
d: 3819173504,	slow: 31077 us,	fast: 19646 us,	speedup: 1.581849

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 4256602993,	slow: 30579 us,	fast: 19171 us,	speedup: 1.595065
d: 2280034797,	slow: 37692 us,	fast: 20424 us,	speedup: 1.845476
d: 2486693976,	slow: 37029 us,	fast: 21143 us,	speedup: 1.751360
d: 2244861190,	slow: 30920 us,	fast: 19197 us,	speedup: 1.610668
d: 2524441561,	slow: 37060 us,	fast: 20538 us,	speedup: 1.804460
d: 3284955087,	slow: 38140 us,	fast: 21367 us,	speedup: 1.784996
d: 2685020394,	slow: 33747 us,	fast: 20915 us,	speedup: 1.613531
d: 3671874172,	slow: 35838 us,	fast: 20545 us,	speedup: 1.744366
d: 3741563917,	slow: 34857 us,	fast: 20607 us,	speedup: 1.691513
d: 3307097346,	slow: 32660 us,	fast: 20298 us,	speedup: 1.609026
```

#### Apple M3

CPU: Apple M3 Pro

OS: macOS 15.1.1

Compiler: Apple clang 16.0.0

```plain
DivBounded, d = rand() + 1
d: 69097088,	slow: 10988 us,	fast: 2545 us,	speedup: 4.317485
d: 1673571830,	slow: 8462 us,	fast: 1801 us,	speedup: 4.698501
d: 2128405245,	slow: 8581 us,	fast: 1426 us,	speedup: 6.017532
d: 1471827830,	slow: 8459 us,	fast: 1633 us,	speedup: 5.180037
d: 146192211,	slow: 9369 us,	fast: 1630 us,	speedup: 5.747853
d: 331181303,	slow: 8437 us,	fast: 1440 us,	speedup: 5.859028
d: 2034013338,	slow: 9077 us,	fast: 1534 us,	speedup: 5.917210
d: 2017462014,	slow: 8979 us,	fast: 1552 us,	speedup: 5.785438
d: 864750009,	slow: 8472 us,	fast: 1437 us,	speedup: 5.895616
d: 1831545208,	slow: 8609 us,	fast: 1420 us,	speedup: 6.062676

DivBounded, d = 2^31
d: 2147483648,	slow: 8681 us,	fast: 1423 us,	speedup: 6.100492

DivBounded, d > 2^31
d: 2147483649,	slow: 8488 us,	fast: 1423 us,	speedup: 5.964863

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3163445217 / 749697952 = 4, DivBounded returns: 0

Div, d = UINT32_MAX - rand()
d: 3408061787,	slow: 8451 us,	fast: 1920 us,	speedup: 4.401562
d: 3758088166,	slow: 8428 us,	fast: 1907 us,	speedup: 4.419507
d: 2546247239,	slow: 8472 us,	fast: 1941 us,	speedup: 4.364760
d: 4018178945,	slow: 8589 us,	fast: 1942 us,	speedup: 4.422760
d: 3762748247,	slow: 8470 us,	fast: 1916 us,	speedup: 4.420668
d: 3558817314,	slow: 8640 us,	fast: 1954 us,	speedup: 4.421699
d: 3475526995,	slow: 8403 us,	fast: 1938 us,	speedup: 4.335913
d: 3774473406,	slow: 8426 us,	fast: 1908 us,	speedup: 4.416143
d: 3055069103,	slow: 8532 us,	fast: 1947 us,	speedup: 4.382126
d: 2359881192,	slow: 8382 us,	fast: 1921 us,	speedup: 4.363352

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 2795184342,	slow: 8401 us,	fast: 1907 us,	speedup: 4.405349
d: 2458441063,	slow: 8416 us,	fast: 1909 us,	speedup: 4.408591
d: 3581044402,	slow: 8475 us,	fast: 1911 us,	speedup: 4.434851
d: 3384040433,	slow: 8421 us,	fast: 2010 us,	speedup: 4.189552
d: 3758117124,	slow: 8540 us,	fast: 2184 us,	speedup: 3.910256
d: 3032944345,	slow: 8420 us,	fast: 2062 us,	speedup: 4.083414
d: 4171228064,	slow: 8519 us,	fast: 2331 us,	speedup: 3.654655
d: 3373882174,	slow: 8367 us,	fast: 2355 us,	speedup: 3.552866
d: 2679466224,	slow: 8334 us,	fast: 2386 us,	speedup: 3.492875
d: 3204216019,	slow: 8543 us,	fast: 2330 us,	speedup: 3.666524
```

### CUDA benchmark

One warp is used to collect the number of cycles for 32 invocations of built-in division, `DivBounded`, and `Div`.
Each case scans ILP=1..8 by generating 1..8 independent dependency chains (`n regs per thread` in the following tables).
"Reference" refers to a reference implementation with built-in integer division.
"Target" refers to the fast division implementation, either `DivBounded` or `Div`.

Conclusions:

1. `DivBounded` achieves a speedup of 2.4x ~ 4.2x;
2. `Div` achieves a speedup of 1.5x ~ 2.5x;
3. `DivBounded` is 1.2 ~ 2.6 times as fast as `Div`.

#### NVIDIA Ampere

GPU: NVIDIA RTX A4000 @ 2.10GHz

OS: Windows 11 23H2

Compiler: MSVC 19.41.34123 + CUDA 12.4.131

```plain
DivBounded
1 warp, 1 regs per thread, #invocation: 32, reference: 1477 cycles,     target: 392 cycles,     speedup: 3.77
1 warp, 2 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 407 cycles,     speedup: 4.27
1 warp, 3 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 407 cycles,     speedup: 4.27
1 warp, 4 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 482 cycles,     speedup: 3.97
1 warp, 5 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 482 cycles,     speedup: 3.97
1 warp, 6 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 639 cycles,     speedup: 3.60
1 warp, 7 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 639 cycles,     speedup: 3.60
1 warp, 8 regs per thread, #invocation: 32, reference: 2555 cycles,     target: 1060 cycles,    speedup: 2.41

Div
1 warp, 1 regs per thread, #invocation: 32, reference: 1478 cycles,     target: 836 cycles,     speedup: 1.77
1 warp, 2 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 799 cycles,     speedup: 2.18
1 warp, 3 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 799 cycles,     speedup: 2.18
1 warp, 4 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 1270 cycles,    speedup: 1.51
1 warp, 5 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 1270 cycles,    speedup: 1.51
1 warp, 6 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 1077 cycles,    speedup: 2.14
1 warp, 7 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 1077 cycles,    speedup: 2.14
1 warp, 8 regs per thread, #invocation: 32, reference: 2555 cycles,     target: 1272 cycles,    speedup: 2.01
```

GPU: NVIDIA RTX A4000 @ 2.10GHz

OS: Ubuntu 22.04.3 LTS (WSL2)

Compiler: GCC 11.4.0 + CUDA 12.6.68

```plain
DivBounded
1 warp, 1 regs per thread, #invocation: 32, reference: 1477 cycles,     target: 393 cycles,     speedup: 3.76
1 warp, 2 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 408 cycles,     speedup: 4.26
1 warp, 3 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 408 cycles,     speedup: 4.26
1 warp, 4 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 487 cycles,     speedup: 3.93
1 warp, 5 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 487 cycles,     speedup: 3.93
1 warp, 6 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 639 cycles,     speedup: 3.60
1 warp, 7 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 639 cycles,     speedup: 3.60
1 warp, 8 regs per thread, #invocation: 32, reference: 2555 cycles,     target: 1060 cycles,    speedup: 2.41

Div
1 warp, 1 regs per thread, #invocation: 32, reference: 1478 cycles,     target: 802 cycles,     speedup: 1.84
1 warp, 2 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 827 cycles,     speedup: 2.10
1 warp, 3 regs per thread, #invocation: 32, reference: 1738 cycles,     target: 827 cycles,     speedup: 2.10
1 warp, 4 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 1270 cycles,    speedup: 1.51
1 warp, 5 regs per thread, #invocation: 32, reference: 1914 cycles,     target: 1270 cycles,    speedup: 1.51
1 warp, 6 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 896 cycles,     speedup: 2.57
1 warp, 7 regs per thread, #invocation: 32, reference: 2301 cycles,     target: 896 cycles,     speedup: 2.57
1 warp, 8 regs per thread, #invocation: 32, reference: 2555 cycles,     target: 1274 cycles,    speedup: 2.01
```

### CUDA results (NOT benchmark)

Compute 2^24 (1 << 24) unsigned integer divisions and check the correctness.
"Reference" refers to a reference implementation with built-in integer division.
"Target" refers to the fast division implementation, either `DivBounded` or `Div`.

The kernels are memory-bound, and the execution time includes both memory access and computation.
So the results do NOT reflect the performance of the fast division, and CANNOT be used as a benchmark.

Since this is a correctness check, only one platform setting is shown here.

GPU: NVIDIA RTX A4000 @ 2.10GHz

OS: Windows 11 23H2

Compiler: MSVC 19.41.34123 + CUDA 12.4.131

```plain
This is a test for correctness, NOT a benchmark.

DivBounded, d = rand() + 1
d: 21232,       reference: 346.11 us,   target: 347.14 us
d: 20746,       reference: 348.16 us,   target: 348.16 us
d: 26458,       reference: 347.14 us,   target: 347.14 us
d: 14591,       reference: 347.14 us,   target: 346.11 us
d: 29445,       reference: 345.09 us,   target: 347.14 us

DivBounded, d = 2^31
d: 2147483648,  reference: 345.09 us,   target: 346.11 us

DivBounded, d > 2^31
d: 4294954493,  reference: 345.09 us,   target: 347.14 us

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3626093760 / 12 = 302174480, target returns: 33739024

Div, d = UINT32_MAX - rand()
d: 4294951337,  reference: 347.14 us,   target: 347.14 us
d: 4294939147,  reference: 347.14 us,   target: 348.03 us
d: 4294953848,  reference: 346.11 us,   target: 347.14 us
d: 4294942284,  reference: 349.22 us,   target: 348.16 us
d: 4294962188,  reference: 347.14 us,   target: 346.05 us

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 4294944099,  reference: 346.11 us,   target: 346.11 us
d: 4294963621,  reference: 346.11 us,   target: 347.14 us
d: 4294951904,  reference: 348.16 us,   target: 346.11 us
d: 4294936770,  reference: 345.12 us,   target: 347.14 us
d: 4294944177,  reference: 346.11 us,   target: 347.14 us
```

### Python results

Compute 2^14 (1 << 14) unsigned integer divisions and check the correctness.

This is just a proof of concept, and one should never expect to accelerate division in interpreted language with this algorithm.

"Reference" refers to a reference implementation with built-in integer division. "Target" refers to the ~~fast~~ division implementation.

CPU: Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz

OS: Windows 11 23H2

Python: 3.10.7

```plain
d <= 2**31, n < 2**31
d: 2141932235,  reference: 783 us,      target: 3304 us
d: 651779455,   reference: 703 us,      target: 3365 us
d: 214010527,   reference: 740 us,      target: 3446 us

d = 2**31, n < 2**31
d: 2147483648,  reference: 566 us,      target: 2999 us

d = 2**31 + 1, n < 2**31
d: 2147483649,  reference: 574 us,      target: 3427 us

d <= 2**31, n >= 2**31
d: 489000381,   reference: 766 us,      target: 3269 us
d: 1488398454,  reference: 1011 us,     target: 3562 us
d: 903438726,   reference: 710 us,      target: 3254 us

d >= 2**31, n < 2**31
d: 2437972628,  reference: 559 us,      target: 3495 us
d: 2620485219,  reference: 570 us,      target: 3438 us
d: 4079159863,  reference: 579 us,      target: 3263 us

d >= 2**31, n >= 2**31
d: 3078107323,  reference: 864 us,      target: 3807 us
d: 4173689700,  reference: 739 us,      target: 3291 us
d: 3872453472,  reference: 701 us,      target: 3514 us
```
