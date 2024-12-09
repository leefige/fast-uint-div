# Fast Unsigned Integer Divsion by Constants

## Experiments

### C++

#### Intel Core

CPU: Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz

OS: Windows 11 23H2

Compiler: MSVC 19.41.34123

```plain
DivBounded, d = rand() + 1
d: 17554,       slow: 2888 us,  fast: 2563 us,  speedup: 1.126805
d: 30065,       slow: 2887 us,  fast: 2500 us,  speedup: 1.154800
d: 18564,       slow: 2502 us,  fast: 3326 us,  speedup: 0.752255
d: 16779,       slow: 3154 us,  fast: 2300 us,  speedup: 1.371304
d: 15830,       slow: 2254 us,  fast: 2023 us,  speedup: 1.114187
d: 24707,       slow: 2293 us,  fast: 2506 us,  speedup: 0.915004
d: 12493,       slow: 2788 us,  fast: 2943 us,  speedup: 0.947333
d: 25092,       slow: 2776 us,  fast: 2745 us,  speedup: 1.011293
d: 12026,       slow: 2250 us,  fast: 2521 us,  speedup: 0.892503
d: 22961,       slow: 2850 us,  fast: 2123 us,  speedup: 1.342440

DivBounded, d = 2^31
d: 2147483648,  slow: 2631 us,  fast: 2727 us,  speedup: 0.964796

This is intended to fail for DivBounded due to d > 2^31
Error: 29648 / 2147483649 = 0, DivBounded returns: 59295

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 4294939206 / 15387 = 279127, DivBounded returns: 16983

Div, d = UINT32_MAX - rand()
d: 4294935273,  slow: 3216 us,  fast: 2940 us,  speedup: 1.093878
d: 4294948086,  slow: 2419 us,  fast: 3456 us,  speedup: 0.699942
d: 4294955517,  slow: 2592 us,  fast: 3375 us,  speedup: 0.768000
d: 4294935807,  slow: 2340 us,  fast: 2463 us,  speedup: 0.950061
d: 4294957770,  slow: 2448 us,  fast: 2684 us,  speedup: 0.912072
d: 4294966715,  slow: 3004 us,  fast: 2562 us,  speedup: 1.172521
d: 4294938026,  slow: 2745 us,  fast: 2486 us,  speedup: 1.104183
d: 4294941306,  slow: 3706 us,  fast: 3260 us,  speedup: 1.136810
d: 4294951306,  slow: 3033 us,  fast: 3023 us,  speedup: 1.003308
d: 4294963678,  slow: 2514 us,  fast: 2574 us,  speedup: 0.976690

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 4294966853,  slow: 3017 us,  fast: 2844 us,  speedup: 1.060830
d: 4294946569,  slow: 2783 us,  fast: 2859 us,  speedup: 0.973417
d: 4294936680,  slow: 2245 us,  fast: 2215 us,  speedup: 1.013544
d: 4294936715,  slow: 3415 us,  fast: 2899 us,  speedup: 1.177992
d: 4294946199,  slow: 2730 us,  fast: 3265 us,  speedup: 0.836141
d: 4294952828,  slow: 2785 us,  fast: 2569 us,  speedup: 1.084079
d: 4294961794,  slow: 2456 us,  fast: 3050 us,  speedup: 0.805246
d: 4294943698,  slow: 2692 us,  fast: 2827 us,  speedup: 0.952246
d: 4294961972,  slow: 2573 us,  fast: 2401 us,  speedup: 1.071637
d: 4294935487,  slow: 2447 us,  fast: 2253 us,  speedup: 1.086107
```

CPU: Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz

OS: Ubuntu 22.04.3 LTS (WSL2)

Compiler: GCC 11.4.0

```plain
DivBounded, d = rand() + 1
d: 527384375,   slow: 4180 us,  fast: 2428 us,  speedup: 1.721582
d: 792346163,   slow: 3928 us,  fast: 2961 us,  speedup: 1.326579
d: 713791299,   slow: 2767 us,  fast: 2720 us,  speedup: 1.017279
d: 478857821,   slow: 3055 us,  fast: 3146 us,  speedup: 0.971074
d: 755374755,   slow: 3647 us,  fast: 5932 us,  speedup: 0.614801
d: 829138282,   slow: 1739 us,  fast: 1713 us,  speedup: 1.015178
d: 515927610,   slow: 3139 us,  fast: 2127 us,  speedup: 1.475787
d: 1467254302,  slow: 2268 us,  fast: 2514 us,  speedup: 0.902148
d: 439758072,   slow: 1938 us,  fast: 2633 us,  speedup: 0.736043
d: 113376739,   slow: 3930 us,  fast: 2290 us,  speedup: 1.716157

DivBounded, d = 2^31
d: 2147483648,  slow: 3113 us,  fast: 3016 us,  speedup: 1.032162

This is intended to fail for DivBounded due to d > 2^31
Error: 1128214007 / 2147483649 = 0, DivBounded returns: 2256428012

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3816214986 / 910300487 = 4, DivBounded returns: 0

Div, d = UINT32_MAX - rand()
d: 2792170072,  slow: 2763 us,  fast: 3165 us,  speedup: 0.872986
d: 3309288323,  slow: 3375 us,  fast: 2894 us,  speedup: 1.166206
d: 3633997112,  slow: 2117 us,  fast: 2446 us,  speedup: 0.865495
d: 3741498301,  slow: 2518 us,  fast: 2762 us,  speedup: 0.911658
d: 3194800613,  slow: 2340 us,  fast: 2021 us,  speedup: 1.157843
d: 3851791869,  slow: 2836 us,  fast: 3571 us,  speedup: 0.794175
d: 3155347687,  slow: 3197 us,  fast: 1985 us,  speedup: 1.610579
d: 2665679419,  slow: 3739 us,  fast: 3072 us,  speedup: 1.217122
d: 3444900026,  slow: 2881 us,  fast: 2275 us,  speedup: 1.266374
d: 2523168788,  slow: 2737 us,  fast: 2763 us,  speedup: 0.990590

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 3965318728,  slow: 2023 us,  fast: 2322 us,  speedup: 0.871232
d: 3395636836,  slow: 4053 us,  fast: 3248 us,  speedup: 1.247845
d: 3135780058,  slow: 3156 us,  fast: 2994 us,  speedup: 1.054108
d: 3404177114,  slow: 2866 us,  fast: 2842 us,  speedup: 1.008445
d: 4053165128,  slow: 3353 us,  fast: 2482 us,  speedup: 1.350927
d: 2866901901,  slow: 2097 us,  fast: 2052 us,  speedup: 1.021930
d: 2239909202,  slow: 2309 us,  fast: 1944 us,  speedup: 1.187757
d: 2731924315,  slow: 2719 us,  fast: 1887 us,  speedup: 1.440911
d: 3877218857,  slow: 3579 us,  fast: 3385 us,  speedup: 1.057312
d: 3631592684,  slow: 3052 us,  fast: 3298 us,  speedup: 0.925409
```

#### Intel Xeon

CPU: Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz

OS: Ubuntu 22.04.2 LTS

Compiler: GCC 11.3.0

```plain
DivBounded, d = rand() + 1
d: 1643827964,	slow: 94 us,	fast: 85 us,	speedup: 1.105882
d: 1297069845,	slow: 67 us,	fast: 83 us,	speedup: 0.807229
d: 588940133,	slow: 77 us,	fast: 89 us,	speedup: 0.865169
d: 798502978,	slow: 107 us,	fast: 68 us,	speedup: 1.573529
d: 1512870593,	slow: 81 us,	fast: 64 us,	speedup: 1.265625
d: 182169805,	slow: 95 us,	fast: 63 us,	speedup: 1.507937
d: 1268708431,	slow: 95 us,	fast: 54 us,	speedup: 1.759259
d: 607005427,	slow: 78 us,	fast: 61 us,	speedup: 1.278689
d: 136306461,	slow: 61 us,	fast: 78 us,	speedup: 0.782051
d: 1894690810,	slow: 75 us,	fast: 75 us,	speedup: 1.000000

DivBounded, d = 2^31
d: 2147483648,	slow: 81 us,	fast: 85 us,	speedup: 0.952941

This is intended to fail for DivBounded due to d > 2^31
Error: 417892282 / 2147483649 = 0, DivBounded returns: 835784563

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 3559143986 / 154830491 = 22, DivBounded returns: 6

Div, d = UINT32_MAX - rand()
d: 2388521870,	slow: 69 us,	fast: 85 us,	speedup: 0.811765
d: 2340684044,	slow: 87 us,	fast: 62 us,	speedup: 1.403226
d: 2321967900,	slow: 80 us,	fast: 105 us,	speedup: 0.761905
d: 3797404963,	slow: 99 us,	fast: 77 us,	speedup: 1.285714
d: 3556239206,	slow: 90 us,	fast: 70 us,	speedup: 1.285714
d: 3035662564,	slow: 93 us,	fast: 98 us,	speedup: 0.948980
d: 3472856481,	slow: 148 us,	fast: 164 us,	speedup: 0.902439
d: 3323759311,	slow: 112 us,	fast: 88 us,	speedup: 1.272727
d: 2299677373,	slow: 157 us,	fast: 130 us,	speedup: 1.207692
d: 3258741941,	slow: 62 us,	fast: 83 us,	speedup: 0.746988

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 2968596582,	slow: 110 us,	fast: 90 us,	speedup: 1.222222
d: 2173783953,	slow: 69 us,	fast: 87 us,	speedup: 0.793103
d: 3331240405,	slow: 61 us,	fast: 92 us,	speedup: 0.663043
d: 3669938107,	slow: 82 us,	fast: 52 us,	speedup: 1.576923
d: 4165233689,	slow: 61 us,	fast: 84 us,	speedup: 0.726190
d: 2686819281,	slow: 81 us,	fast: 75 us,	speedup: 1.080000
d: 2373950908,	slow: 70 us,	fast: 77 us,	speedup: 0.909091
d: 4006521815,	slow: 100 us,	fast: 96 us,	speedup: 1.041667
d: 2231217960,	slow: 79 us,	fast: 73 us,	speedup: 1.082192
d: 3873967956,	slow: 92 us,	fast: 82 us,	speedup: 1.121951
```

#### Apple M3

CPU: Apple M3 Pro

OS: macOS 15.1.1

Compiler: Apple clang 16.0.0

```plain
DivBounded, d = rand() + 1
d: 1746407746,	slow: 179 us,	fast: 197 us,	speedup: 0.908629
d: 1053666375,	slow: 160 us,	fast: 285 us,	speedup: 0.561404
d: 987025081,	slow: 180 us,	fast: 276 us,	speedup: 0.652174
d: 1642338126,	slow: 213 us,	fast: 176 us,	speedup: 1.210227
d: 2051586970,	slow: 99 us,	fast: 105 us,	speedup: 0.942857
d: 32029009,	slow: 220 us,	fast: 159 us,	speedup: 1.383648
d: 1987402252,	slow: 170 us,	fast: 219 us,	speedup: 0.776256
d: 2054970499,	slow: 242 us,	fast: 181 us,	speedup: 1.337017
d: 1732404862,	slow: 225 us,	fast: 122 us,	speedup: 1.844262
d: 332795599,	slow: 272 us,	fast: 178 us,	speedup: 1.528090

DivBounded, d = 2^31
d: 2147483648,	slow: 197 us,	fast: 250 us,	speedup: 0.788000

This is intended to fail for DivBounded due to d > 2^31
Error: 304426472 / 2147483649 = 0, DivBounded returns: 608852943

This is highly probable to fail for DivBounded due to n >= 2^31
Error: 2621846615 / 1189667751 = 2, DivBounded returns: 0

Div, d = UINT32_MAX - rand()
d: 3306572353,	slow: 229 us,	fast: 180 us,	speedup: 1.272222
d: 4185477833,	slow: 174 us,	fast: 107 us,	speedup: 1.626168
d: 4192195896,	slow: 220 us,	fast: 164 us,	speedup: 1.341463
d: 4269316123,	slow: 280 us,	fast: 280 us,	speedup: 1.000000
d: 3760716723,	slow: 238 us,	fast: 200 us,	speedup: 1.190000
d: 3558556935,	slow: 181 us,	fast: 183 us,	speedup: 0.989071
d: 2642648744,	slow: 273 us,	fast: 123 us,	speedup: 2.219512
d: 3615555453,	slow: 126 us,	fast: 141 us,	speedup: 0.893617
d: 4056477701,	slow: 155 us,	fast: 163 us,	speedup: 0.950920
d: 2170919561,	slow: 190 us,	fast: 175 us,	speedup: 1.085714

Div, d = UINT32_MAX - rand(), n = UINT32_MAX - rand()
d: 2311650892,	slow: 195 us,	fast: 166 us,	speedup: 1.174699
d: 3190282830,	slow: 181 us,	fast: 151 us,	speedup: 1.198675
d: 2435519087,	slow: 107 us,	fast: 142 us,	speedup: 0.753521
d: 2792514523,	slow: 243 us,	fast: 186 us,	speedup: 1.306452
d: 3412270285,	slow: 107 us,	fast: 199 us,	speedup: 0.537688
d: 3499961325,	slow: 212 us,	fast: 140 us,	speedup: 1.514286
d: 3893779290,	slow: 335 us,	fast: 305 us,	speedup: 1.098361
d: 3208412709,	slow: 318 us,	fast: 323 us,	speedup: 0.984520
d: 3785995684,	slow: 288 us,	fast: 254 us,	speedup: 1.133858
d: 3904381871,	slow: 320 us,	fast: 292 us,	speedup: 1.095890
```
