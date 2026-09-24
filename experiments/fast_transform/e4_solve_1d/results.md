# E4: end-to-end 1D rootfinding with the dense, fast and auto transforms

Branch `fast-transform`. Nothing under `yroots/` or `tests/` was changed. Environment: Python 3.14.0 free-threaded build, finufft 2.5.1, numpy 2.4.6, scipy 1.18.1, macOS, 12 cores shared with other agents' benchmarks. Every solve used `max_cpu=1` and `FastTransform.NUFFT_NTHREADS=1`, with `NUFFT_EPS=1e-15` (the default). Each (problem, method) pair got one warm-up solve that is not timed. After that the time is the median of 5 repeats (3 when a solve took more than 3 s; when the warm-up took more than 15 s, the warm-up plus one repeat). Then one extra instrumented solve wrapped `ChebyshevSubdivisionSolver.TransformChebInPlaceND` to record the degree of each call, whether it went through the NUFFT, and how long it took. **Timing noise:** runs that make exactly the same transform calls differ by about ±15–20% (for example the dense-only configurations of `sin_100x` measured 0.025–0.030 s), so gaps under about 20% mean nothing.

Methods:
- `dense`: the default path (`TRANSFORM_METHOD='dense'`).
- `fast`: `TRANSFORM_METHOD='fast'`, so every call uses the NUFFT when `|alpha|+|beta|<=1`.
- `autoN`: `TRANSFORM_METHOD='auto'`, `FAST_TRANSFORM_NDIMS=(1,)`, `FAST_TRANSFORM_MIN_DEGREE=N`, for N in {64, 128, 256, 512, 1024, 2048, 4096}.
- `fastT` and `autoTN` (a prototype that exists only in the experiment script, `_truncT` in `e4_bench.py`): the same as `fast`/`autoN`, except that after each NUFFT call the trailing rows are dropped while their cumulative absolute sum stays at or below the transformation error the solver already budgets for that call (`n * 2^-52 * absSum(M)`, from `getTransformationError`). The file `e4_results_chop0.01.csv` holds a stricter version that may use only 1% of that budget.

Reference roots: closed form for sin, cos(ωx²), T_n and the polynomials; `scipy.special.jn_zeros` for J0. For exp(x)−cos(ωx) the reference is independent of yroots: sign changes on an 8,000,001-point grid, then `brentq`, then `mpmath.findroot` at 40 digits. A computed root counts as matched when it lies within 1e-7·max(1,|a|,|b|) of a reference root. Errors are relative to that same scale.

## Per-problem results

`deg` is the Chebyshev degree `chebApproximate` chose on [a,b] (for MultiCheb/MultiPower inputs it is the coefficient degree). Times are median solve seconds. `dense_trans_share` is the fraction of the instrumented dense solve spent inside `TransformChebInPlaceND`. `root_sets_ok` means every one of the 13 methods found exactly the reference root set, with no missing, spurious or duplicate roots. `err_worst` is the worst max-relative root error over all 13 methods.

| problem | deg | roots | dense | fast | auto256 | auto512 | auto1024 | auto2048 | auto4096 | autoT1024 | dense_trans_share | root_sets_ok | err_dense | err_worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poly3_callable | 3 | 3/3 | 0.00167 | 0.0146 | 0.00151 | 0.00149 | 0.00149 | 0.00132 | 0.00152 | 0.00153 | 2.0% | yes | 2.2e-16 | 7.9e-16 |
| poly4_MultiPower | 4 | 4/4 | 0.00179 | 0.0198 | 0.00183 | 0.00179 | 0.00185 | 0.00178 | 0.00179 | 0.00181 | 2.4% | yes | 2.2e-16 | 5.8e-16 |
| sin_10x | 33 | 7/7 | 0.00325 | 0.0311 | 0.00332 | 0.00326 | 0.00323 | 0.00329 | 0.00329 | 0.00349 | 2.2% | yes | 1.1e-16 | 1.0e-15 |
| exp_minus_cos30x | 62 | 10/10 | 0.00487 | 0.0515 | 0.00495 | 0.00495 | 0.00494 | 0.00494 | 0.00493 | 0.00497 | 2.7% | yes | 8.4e-16 | 3.7e-14 |
| exp_minus_cos30x_[-3,0.5] | 92 | 30/30 | 0.0123 | 0.133 | 0.0124 | 0.0124 | 0.0124 | 0.0124 | 0.0124 | 0.0131 | 2.7% | yes | 1.4e-15 | 3.1e-14 |
| T100_MultiCheb | 100 | 100/100 | 0.0373 | 0.409 | 0.0372 | 0.0373 | 0.0372 | 0.037 | 0.0373 | 0.0378 | 2.7% | yes | 2.8e-16 | 6.7e-16 |
| sin_100x | 147 | 63/63 | 0.03 | 0.281 | 0.0305 | 0.0303 | 0.0247 | 0.0246 | 0.0246 | 0.0292 | 2.8% | yes | 2.2e-16 | 1.4e-15 |
| sin100x_times_x-0.3 | 148 | 64/64 | 0.0251 | 0.276 | 0.0253 | 0.0252 | 0.0252 | 0.0251 | 0.0252 | 0.0255 | 2.8% | yes | 8.3e-16 | 9.7e-14 |
| cos_100x2 | 176 | 64/64 | 0.0249 | 0.274 | 0.025 | 0.0249 | 0.0249 | 0.0249 | 0.0249 | 0.025 | 2.9% | yes | 2.2e-16 | 1.6e-15 |
| T300_callable | 300 | 300/300 | 0.118 | 1.29 | 0.236 | 0.115 | 0.123 | 0.123 | 0.118 | 0.12 | 2.8% | yes | 4.4e-16 | 1.6e-15 |
| J0_300x | 366 | 190/190 | 0.0719 | 0.806 | 0.18 | 0.072 | 0.0738 | 0.0727 | 0.0724 | 0.0729 | 3.1% | yes | 3.3e-16 | 3.0e-15 |
| exp_minus_cos300x | 366 | 96/96 | 0.0486 | 0.55 | 0.145 | 0.0484 | 0.0487 | 0.0496 | 0.0487 | 0.049 | 3.6% | yes | 7.9e-16 | 4.1e-14 |
| T1000_MultiCheb | 1000 | 1000/1000 | 0.378 | 4.45 | 1.52 | 1.2 | 0.42 | 0.426 | 0.409 | 0.421 | 3.8% | yes | 4.4e-16 | 2.0e-15 |
| J0_x_[0,2000] | 1094 | 636/636 | 0.235 | 2.52 | 0.473 | 0.417 | 0.297 | 0.246 | 0.235 | 0.249 | 4.1% | yes | 2.3e-16 | 8.4e-15 |
| sin_1000x | 1097 | 637/637 | 0.286 | 3.2 | 0.94 | 0.787 | 0.514 | 0.293 | 0.289 | 0.311 | 4.2% | yes | 3.9e-16 | 2.1e-15 |
| cos_1000x2 | 1156 | 636/636 | 0.245 | 2.83 | 0.866 | 0.696 | 0.475 | 0.245 | 0.251 | 0.25 | 4.3% | yes | 4.4e-16 | 6.7e-15 |
| sin_x_[1,3000] | 1608 | 954/954 | 0.365 | 4.13 | 1.22 | 1.08 | 0.91 | 0.363 | 0.363 | 0.365 | 4.9% | yes | 3.0e-16 | 1.9e-15 |
| J0_3000x | 3134 | 1910/1910 | 0.752 | 8.16 | 2.03 | 1.9 | 1.75 | 1.62 | 0.783 | 0.748 | 6.5% | yes | 3.3e-16 | 5.1e-15 |
| T4000_MultiCheb | 4000 | 4000/4000 | 1.79 | 19.9 | 6.02 | 5.41 | 4.95 | 4.64 | 1.62 | 1.59 | 10.3% | yes | 4.4e-16 | 4.6e-15 |
| sin_x_[-2000,6000] | 4150 | 2546/2546 | 1.01 | 11.1 | 3.21 | 3.09 | 2.76 | 2.59 | 1.87 | 0.979 | 7.8% | yes | 3.0e-16 | 3.9e-15 |
| sin_5000x | 5159 | 3183/3183 | 1.5 | 16.1 | 4.66 | 4.37 | 4.17 | 4.01 | 6.41 | 1.39 | 12.2% | yes | 4.4e-16 | 5.3e-15 |
| cos_5000x2 | 5252 | 3184/3184 | 1.37 | 14 | 3.87 | 3.57 | 3.4 | 3.3 | 4.42 | 1.28 | 14.6% | yes | 4.4e-16 | 8.1e-15 |
| sin_20000x | 20245 | 12733/12733 | 6.77 | 62.4 | 19 | 17.5 | 17.2 | 17.6 | 20 | 5 | 22.8% | yes | 5.0e-16 | 8.9e-15 |

Every method, in seconds (`e4_summary_table_all_methods.md`):

| problem | deg | dense | fast | auto64 | auto128 | auto256 | auto512 | auto1024 | auto2048 | auto4096 | fastT | autoT512 | autoT1024 | autoT2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poly3_callable | 3 | 0.00167 | 0.0146 | 0.00151 | 0.0015 | 0.00151 | 0.00149 | 0.00149 | 0.00132 | 0.00152 | 0.0149 | 0.00153 | 0.00153 | 0.00154 |
| poly4_MultiPower | 4 | 0.00179 | 0.0198 | 0.00178 | 0.00181 | 0.00183 | 0.00179 | 0.00185 | 0.00178 | 0.00179 | 0.0199 | 0.00182 | 0.00181 | 0.00181 |
| sin_10x | 33 | 0.00325 | 0.0311 | 0.00326 | 0.00326 | 0.00332 | 0.00326 | 0.00323 | 0.00329 | 0.00329 | 0.0315 | 0.00344 | 0.00349 | 0.0035 |
| exp_minus_cos30x | 62 | 0.00487 | 0.0515 | 0.00477 | 0.00496 | 0.00495 | 0.00495 | 0.00494 | 0.00494 | 0.00493 | 0.0507 | 0.00502 | 0.00497 | 0.00497 |
| exp_minus_cos30x_[-3,0.5] | 92 | 0.0123 | 0.133 | 0.106 | 0.0124 | 0.0124 | 0.0124 | 0.0124 | 0.0124 | 0.0124 | 0.134 | 0.0131 | 0.0131 | 0.013 |
| T100_MultiCheb | 100 | 0.0373 | 0.409 | 0.337 | 0.037 | 0.0372 | 0.0373 | 0.0372 | 0.037 | 0.0373 | 0.394 | 0.038 | 0.0378 | 0.0374 |
| sin_100x | 147 | 0.03 | 0.281 | 0.241 | 0.0624 | 0.0305 | 0.0303 | 0.0247 | 0.0246 | 0.0246 | 0.26 | 0.0296 | 0.0292 | 0.0298 |
| sin100x_times_x-0.3 | 148 | 0.0251 | 0.276 | 0.201 | 0.0546 | 0.0253 | 0.0252 | 0.0252 | 0.0251 | 0.0252 | 0.264 | 0.0255 | 0.0255 | 0.0257 |
| cos_100x2 | 176 | 0.0249 | 0.274 | 0.254 | 0.16 | 0.025 | 0.0249 | 0.0249 | 0.0249 | 0.0249 | 0.258 | 0.0258 | 0.025 | 0.0255 |
| T300_callable | 300 | 0.118 | 1.29 | 1.07 | 0.611 | 0.236 | 0.115 | 0.123 | 0.123 | 0.118 | 1.17 | 0.12 | 0.12 | 0.12 |
| J0_300x | 366 | 0.0719 | 0.806 | 0.511 | 0.323 | 0.18 | 0.072 | 0.0738 | 0.0727 | 0.0724 | 0.738 | 0.0732 | 0.0729 | 0.0727 |
| exp_minus_cos300x | 366 | 0.0486 | 0.55 | 0.448 | 0.325 | 0.145 | 0.0484 | 0.0487 | 0.0496 | 0.0487 | 0.496 | 0.0491 | 0.049 | 0.049 |
| T1000_MultiCheb | 1000 | 0.378 | 4.45 | 2.47 | 1.86 | 1.52 | 1.2 | 0.42 | 0.426 | 0.409 | 4.21 | 0.433 | 0.421 | 0.457 |
| J0_x_[0,2000] | 1094 | 0.235 | 2.52 | 0.633 | 0.557 | 0.473 | 0.417 | 0.297 | 0.246 | 0.235 | 2.42 | 0.24 | 0.249 | 0.249 |
| sin_1000x | 1097 | 0.286 | 3.2 | 1.7 | 1.28 | 0.94 | 0.787 | 0.514 | 0.293 | 0.289 | 2.71 | 0.301 | 0.311 | 0.298 |
| cos_1000x2 | 1156 | 0.245 | 2.83 | 1.38 | 1.04 | 0.866 | 0.696 | 0.475 | 0.245 | 0.251 | 2.47 | 0.255 | 0.25 | 0.252 |
| sin_x_[1,3000] | 1608 | 0.365 | 4.13 | 1.87 | 1.46 | 1.22 | 1.08 | 0.91 | 0.363 | 0.363 | 3.55 | 0.369 | 0.365 | 0.367 |
| J0_3000x | 3134 | 0.752 | 8.16 | 2.61 | 2.27 | 2.03 | 1.9 | 1.75 | 1.62 | 0.783 | 7.29 | 0.744 | 0.748 | 0.746 |
| T4000_MultiCheb | 4000 | 1.79 | 19.9 | 7.72 | 6.59 | 6.02 | 5.41 | 4.95 | 4.64 | 1.62 | 15.4 | 1.6 | 1.59 | 1.6 |
| sin_x_[-2000,6000] | 4150 | 1.01 | 11.1 | 3.94 | 3.59 | 3.21 | 3.09 | 2.76 | 2.59 | 1.87 | 9.23 | 0.973 | 0.979 | 0.997 |
| sin_5000x | 5159 | 1.5 | 16.1 | 5.96 | 5.08 | 4.66 | 4.37 | 4.17 | 4.01 | 6.41 | 14.1 | 1.29 | 1.39 | 1.44 |
| cos_5000x2 | 5252 | 1.37 | 14 | 4.89 | 4.28 | 3.87 | 3.57 | 3.4 | 3.3 | 4.42 | 12.2 | 1.27 | 1.28 | 1.29 |
| sin_20000x | 20245 | 6.77 | 62.4 | 21.8 | 19.7 | 19 | 17.5 | 17.2 | 17.6 | 20 | 48.2 | 5.03 | 5 | 5.06 |

### Root sets and accuracy
- **Every method found exactly the reference root set on all 23 problems**: 23 × 13 = 299 solves, with no missing, spurious or duplicate roots. That covers 12,733 roots for sin(20000x), 4000 for T_4000, J0 on [0,2000], sin(x) on [-2000,6000], and so on. Every method returned the same number of roots as dense. Its roots differed from dense by at most 3.2e-14 (relative) for plain `fast`/`auto` and 9.8e-14 for the tail-chop prototype.
- Accuracy: dense gets max relative error 1e-16 to 1.4e-15. Plain `fast`/`auto` get ≤1e-15 on most problems and up to 3e-14 on exp(x)−cos(300x); that problem's reference roots include a nearly double pair next to x=0, and a max error of 3e-14 is still near the conditioning limit. The tail-chop prototype makes the error 10–20× larger on the high-degree problems (roughly 5e-15 to 9e-15, against 5e-16 for dense), because it spends the full per-transform error budget. With a 1% budget the error returns to about 1e-15, but the speedup disappears (see below).

## Why `fast` / `auto` are slower end to end

1. **The NUFFT has a fixed cost of about 0.5 ms per call.** Standalone timing (`e4_tail_diag.txt`, 1 thread):

| degree | alpha | dense (µs) | NUFFT (µs) | dense/NUFFT |
|---|---|---|---|---|
| 256 | 0.95 | 54 | 626 | 0.09 |
| 1024 | 0.5 / 0.95 | 645 / 829 | 798 / 801 | 0.81 / 1.04 |
| 2048 | 0.5 / 0.95 | 2486 / 3280 | 1034 / 1057 | 2.4 / 3.1 |
| 4096 | 0.95 | 12260 | 1637 | 7.5 |
| 16384 | 0.95 | 197370 | 4898 | 40 |

   So a single call breaks even at about **degree 1000**. The 1D subdivision solver, though, makes thousands of calls, and most are at degree below 64 (e.g. sin(20000x) makes 92k calls). At those degrees the dense transform takes 1–2 µs, so each NUFFT call is about 300–500× more expensive. Always-`fast` is therefore about 10× slower on every problem, degree 3 as well as degree 20245.

2. **The NUFFT output keeps its full length, and this is what defeats `auto`.** The dense `TransformChebInPlace1D` stops growing its output as soon as the newest diagonal entry of the transform matrix is ≤1e-16 (roughly |alpha|^k ≤ 1e-16). That shortens the result on every zoom or subdivision: a degree-1097 sin(1000x) approximation mapped onto a subinterval of half-width 0.5, 0.1, 0.02 or 0.001 comes back with 627, 161, 53 or 15 coefficients. The NUFFT path returns all n+1 coefficients, and the tail is floored at about 1e-14 (roughly eps·‖p‖). That is far above `trimMs`'s threshold (1e-3 × the approximation error, about 4e-17 here), so the degree never comes down. Every child interval then inherits the full degree, which forces more high-degree calls further down. The count of transform calls at degree ≥256 shows it:

| problem | deg | dense | fast | auto64 | auto128 | auto256 | auto512 | auto1024 | auto2048 | auto4096 | fastT | autoT512 | autoT1024 | autoT2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T300_callable | 300 | 2 | 154 | 154 | 154 | 154 | 2 | 2 | 2 | 2 | 6 | 2 | 2 | 2 |
| J0_300x | 366 | 4 | 140 | 140 | 140 | 140 | 4 | 4 | 4 | 4 | 2 | 4 | 4 | 4 |
| exp_minus_cos300x | 366 | 4 | 139 | 139 | 139 | 139 | 4 | 4 | 4 | 4 | 2 | 4 | 4 | 4 |
| T1000_MultiCheb | 1000 | 26 | 1106 | 1106 | 1106 | 1106 | 1094 | 26 | 26 | 26 | 22 | 28 | 26 | 26 |
| J0_x_[0,2000] | 1094 | 20 | 258 | 258 | 258 | 258 | 258 | 102 | 20 | 20 | 22 | 30 | 22 | 20 |
| sin_1000x | 1097 | 18 | 602 | 602 | 602 | 602 | 578 | 348 | 18 | 18 | 14 | 22 | 22 | 18 |
| cos_1000x2 | 1156 | 22 | 619 | 619 | 619 | 619 | 607 | 355 | 22 | 22 | 20 | 32 | 38 | 22 |
| sin_x_[1,3000] | 1608 | 28 | 744 | 744 | 744 | 744 | 744 | 694 | 28 | 28 | 22 | 30 | 42 | 28 |
| J0_3000x | 3134 | 52 | 903 | 903 | 903 | 903 | 896 | 839 | 756 | 52 | 60 | 68 | 76 | 70 |
| T4000_MultiCheb | 4000 | 100 | 2514 | 2514 | 2514 | 2514 | 2512 | 2438 | 2378 | 100 | 130 | 162 | 178 | 180 |
| sin_x_[-2000,6000] | 4150 | 78 | 1286 | 1286 | 1286 | 1286 | 1286 | 1240 | 1188 | 420 | 68 | 92 | 120 | 116 |
| sin_5000x | 5159 | 88 | 1644 | 1644 | 1644 | 1644 | 1644 | 1560 | 1540 | 1228 | 94 | 126 | 142 | 136 |
| cos_5000x2 | 5252 | 102 | 1538 | 1538 | 1538 | 1538 | 1538 | 1510 | 1484 | 1256 | 108 | 138 | 166 | 174 |
| sin_20000x | 20245 | 342 | 4234 | 4234 | 4234 | 4234 | 4226 | 4200 | 4194 | 4078 | 360 | 422 | 430 | 422 |

   With `auto` at any threshold at or below the problem degree, the number of calls at degree ≥256 grows 10–50×, and most of those become NUFFT calls at about 0.6–1 ms each. `auto4096` on sin(5000x) is **slower** than `auto2048` (6.4 s against 4.0 s). The one top-level NUFFT call returns an untruncated degree-5159 array, and the dense path then transforms it at full degree over and over.

3. **Even a perfect transform could only save a little (Amdahl's law).** With dense, the transform is 2–5% of 1D solve time up to degree about 1600, 6–15% at degree 3000–5000, and 23% at degree 20245. The rest is approximation, bounding and bookkeeping. The best possible end-to-end gain is therefore under 5% below degree 2000 and about 23% at degree 20000.

## Degree at which there is an end-to-end speedup

- **`fast`**: none. It is 9–12× slower at every degree from 3 to 20245.
- **`auto`, as implemented, with FAST_TRANSFORM_MIN_DEGREE in {64, 128, 256, 512, 1024, 2048, 4096}**: none. Once the threshold is at or below the problem's degree, `auto` is 1.2–4× slower. It is only neutral (within noise) when the threshold is above every degree the solve reaches, which in practice means the NUFFT is never called.
- **Prototype `auto` + tail chop (autoT512/1024/2048)**: roughly break-even up to degree about 3000 (within noise). It gives a real gain at degree ≥4000: T_4000 −11%, sin(x) on [-2000,6000] (deg 4150) −3%, sin(5000x) −7 to −14%, cos(5000x²) −6%, sin(20000x) −26% (6.77 s → 5.00 s). That is close to the Amdahl ceiling, and the root sets are unchanged. The cost is about 10× larger root error (still below 1e-14). A 1% chop budget brings the degree inflation back and makes it about 2× slower than dense again (`e4_results_chop0.01.csv`).

## Recommendation for FAST_TRANSFORM_MIN_DEGREE in 1D

- **As `FastTransform.py` and `useFastTransform` stand: do not enable the NUFFT for 1D.** Keep `TRANSFORM_METHOD='dense'`, or equivalently set the 1D `FAST_TRANSFORM_MIN_DEGREE` above any degree you expect (≥ 32768 in practice). The current default of 256 makes 1D solves 1.2–4× slower for every problem of degree ≥256 in this suite, without changing roots.
- **If the fast path is changed so that it truncates its output** the way the dense transform does (drop trailing rows below the transform-error or noise floor; ideally treat that loss properly in the error bound), then **FAST_TRANSFORM_MIN_DEGREE = 1024** is the right 1D threshold. That is the single-call break-even point. In end-to-end tests 512, 1024 and 2048 were all within noise of each other; below 512 the 0.5 ms fixed cost dominates. Expect gains only at degree ≥ about 4000, up to about 25% at degree about 20000.
- Cutting the fixed NUFFT overhead (about 0.5 ms per call, likely FINUFFT type-3 plan setup; for example by reusing plans or batching the two `chebTransform1D` calls made per subdivision) would lower the break-even point. It still would not help the thousands of calls at degree below 64.

## Test suite with `TRANSFORM_METHOD='fast'` forced

Runner: `run_tests_fast.py` sets `C.TRANSFORM_METHOD` and calls `pytest.main([...])`. `tests/` was not edited.

- Command as requested, with `-x`: **1 failed, 166 passed** (stopped at the first failure, `test_Combined_Solver.py::test_multiCheb_multiPower_non_unit_box`). Log: `pytest_fast_x.log`.
- Full run without `-x`: the xfail test `test_known_failures.py::test_ill_conditioned_system_keeps_its_root_at_1e_10` did not finish (over 14 CPU-minutes). It is expected to recurse until RecursionError, and at about 0.5 ms per NUFFT call on degree-1 polynomials that takes a very long time. So the three RecursionError xfail tests were deselected. **Result: 12 failed, 540 passed, 1 xfailed, 3 deselected** (`pytest_fast_full.log`).
- Baseline with the same deselection, `dense`: **552 passed, 1 xfailed, 3 deselected** (`pytest_dense_full.log`). Default `auto` (threshold 256, 1D only), with nothing deselected: **552 passed, 4 xfailed** (`pytest_auto_full.log`).

The 12 failures under forced `fast` are all 2D systems. `auto` never touches them, because FAST_TRANSFORM_NDIMS=(1,).

| test | failure | explanation |
|---|---|---|
| test_Combined_Solver::test_multiCheb_multiPower_non_unit_box, ::test_multiCheb, ::test_outside_neg1_pos1 | residual 1.2–1.6e-14 against a 1e-14 tolerance | NUFFT noise of about eps·‖p‖ per transform; the roots are correct, just slightly less accurate |
| test_chebfun2_suite 7.2 (vanish), 3.1 (polished ref), 4.2 (vanish) | residual or distance 1.8–2× over tight tolerances | same cause |
| test_chebfun2_suite 2.4 (3 tests) | RecursionError | the noise tail stops the degree from shrinking and the boxes from separating, so subdivision never stops |
| test_chebfun2_suite 6.1 (2 tests) | 5 roots returned, 6 expected (the problem has a double root) | the noise floor changes how the double root is resolved |
| test_solve_api::test_finds_all_roots_of_a_system_with_known_solutions | returned 1080 roots instead of 3 | all 1080 are within 2e-15 of the 3 true roots, i.e. duplicates. The untruncated noisy tails keep the final-step boxes from shrinking and merging (the "Might Have Duplicate Roots" warning appears) |

## Files

- `e4_bench.py`: the benchmark (problem suite, instrumentation, tail-chop prototype). `e4_plot.py`: figures and tables. `e4_tail_diag.py`: coefficient-tail diagnostic and single-call timing. `run_tests_fast.py`: pytest runner with the method forced.
- `e4_results.csv`: raw per-(problem, method) results (timings, transform share, call counts, degree statistics, root agreement). `e4_transform_calls.csv`: per-(problem, method, degree bin) call counts, NUFFT counts and mean µs per call. `e4_results_chop0.01.csv` and `e4_transform_calls_chop0.01.csv`: the 1%-budget chop run. `e4_refs.npz`: reference roots. `e4_bench.log`: console log.
- `e4_solve_time_vs_degree.png` and `e4_speedup_vs_degree.png`: figures.
- `e4_summary_table*.md`, `e4_calls_ge256_table.md`, `e4_fast_calls_table.md`: generated tables. `e4_tail_diag.txt`: diagnostic output.
- `pytest_*.log`: test-suite logs.
