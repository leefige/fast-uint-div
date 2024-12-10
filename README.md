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

```

#### Apple M3

CPU: Apple M3 Pro

OS: macOS 15.1.1

Compiler: Apple clang 16.0.0

```plain

```

### CUDA results (not benchmark)

#### NVIDIA Ampere

GPU: NVIDIA RTX A4000 @ 2.10GHz

OS: Windows 11 23H2

Compiler: MSVC 19.41.34123 + CUDA 12.4.131

```plain
This is a test for correctness, NOT a benchmark.

DivBounded, d = rand() + 1
d: 13486,       reference: 346.98 us,   target: 347.14 us
d: 4916,        reference: 347.14 us,   target: 346.11 us
d: 1287,        reference: 348.16 us,   target: 346.11 us
d: 9357,        reference: 347.14 us,   target: 348.16 us
d: 20333,       reference: 347.14 us,   target: 347.14 us

DivBounded, d = 2^31
d: 2147483648,  reference: 348.16 us,   target: 347.14 us

DivBounded, d > 2^31
d: 4294952665,  reference: 346.11 us,   target: 346.11 us

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3626093760 / 26841 = 135095, target returns: 4023

Div, d = UINT32_MAX - rand()
d: 4294946994,  reference: 346.11 us,   target: 348.16 us
d: 4294949189,  reference: 346.11 us,   target: 346.11 us
d: 4294935102,  reference: 346.11 us,   target: 348.16 us
d: 4294940746,  reference: 346.11 us,   target: 347.14 us
d: 4294965264,  reference: 345.09 us,   target: 347.14 us
d: 4294956174,  reference: 346.11 us,   target: 346.11 us
d: 4294960887,  reference: 345.09 us,   target: 348.16 us
d: 4294944288,  reference: 347.14 us,   target: 347.14 us
d: 4294966091,  reference: 346.11 us,   target: 348.16 us
d: 4294943334,  reference: 347.14 us,   target: 346.11 us

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 4294958478,  reference: 370.69 us,   target: 346.11 us
d: 4294962618,  reference: 344.06 us,   target: 346.11 us
d: 4294950685,  reference: 345.09 us,   target: 347.14 us
d: 4294947025,  reference: 347.14 us,   target: 347.14 us
d: 4294944616,  reference: 347.01 us,   target: 348.16 us
d: 4294944925,  reference: 346.11 us,   target: 347.14 us
d: 4294958567,  reference: 347.14 us,   target: 346.11 us
d: 4294941661,  reference: 347.14 us,   target: 348.16 us
d: 4294962893,  reference: 346.11 us,   target: 347.14 us
d: 4294941852,  reference: 347.14 us,   target: 346.11 us
```

GPU: NVIDIA RTX A4000 @ 2.10GHz

OS: Ubuntu 22.04.3 LTS (WSL2)

Compiler: GCC 11.4.0 + CUDA 12.6.68

```plain
This is a test for correctness, NOT a benchmark.

DivBounded, d = rand() + 1
d: 2078777849,  reference: 347.14 us,   target: 348.16 us
d: 402477966,   reference: 346.11 us,   target: 347.14 us
d: 1121148636,  reference: 345.09 us,   target: 346.11 us
d: 1443389175,  reference: 347.14 us,   target: 348.16 us
d: 1673025871,  reference: 347.14 us,   target: 347.14 us

DivBounded, d = 2^31
d: 2147483648,  reference: 347.14 us,   target: 347.14 us

DivBounded, d > 2^31
d: 3382218489,  reference: 346.11 us,   target: 347.14 us

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 4145580802 / 1848127918 = 2, target returns: 0

Div, d = UINT32_MAX - rand()
d: 3567593624,  reference: 347.14 us,   target: 347.10 us
d: 2646522883,  reference: 346.11 us,   target: 347.14 us
d: 2363148066,  reference: 347.14 us,   target: 347.14 us
d: 3847857914,  reference: 347.14 us,   target: 348.16 us
d: 2944589833,  reference: 347.14 us,   target: 348.16 us
d: 3929178683,  reference: 346.24 us,   target: 347.14 us
d: 4151707612,  reference: 347.14 us,   target: 346.11 us
d: 2357112109,  reference: 347.14 us,   target: 348.16 us
d: 3209624395,  reference: 347.14 us,   target: 347.14 us
d: 2712703575,  reference: 346.11 us,   target: 346.11 us

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 2236890087,  reference: 346.11 us,   target: 348.16 us
d: 4184200621,  reference: 346.11 us,   target: 347.14 us
d: 3301258056,  reference: 346.11 us,   target: 348.16 us
d: 2900419013,  reference: 347.14 us,   target: 347.14 us
d: 2997091688,  reference: 348.16 us,   target: 347.14 us
d: 4012162948,  reference: 347.14 us,   target: 347.14 us
d: 2363263881,  reference: 346.11 us,   target: 346.11 us
d: 3703346868,  reference: 347.14 us,   target: 346.27 us
d: 4259161815,  reference: 346.11 us,   target: 347.14 us
d: 3981441505,  reference: 346.11 us,   target: 347.14 us
```
