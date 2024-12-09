# Fast Unsigned Integer Divsion by Constants

## Experiments

### C++

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
