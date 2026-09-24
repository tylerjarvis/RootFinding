# E5 — End-to-end rootfinding in 2D and 5D: dense vs fast (NUFFT) vs auto transforms

Branch `fast-transform`; no files under `yroots/` or `tests/` modified. All runs single-threaded
(`max_cpu=1`, `NUFFT_NTHREADS=1`, `OMP/NUMBA_NUM_THREADS=1`), in `uv run --no-sync python` (CPython 3.14t,
free-threaded), on a shared 12-core Mac with other benchmarks running concurrently. Each problem was solved once
(dense) to warm up numba JIT, then once per method *instrumented* (transform wrapper: per-call time,
degree, ndim, fast/dense path), then timed *un-instrumented* in round-robin order; the table reports the median
(7 reps if <0.5 s, 5 if <5 s, 3 if <30 s, otherwise 1). Solves that fail are recorded as failures; runs are cut
off at max(120 s, 25x the dense warm-up time).

Methods: `dense` (default). `fast` = NUFFT whenever |alpha|+|beta|<=1. `auto_ndX_dY` = NUFFT only when ndim in X and
degree >= Y (nd12 = ndims (1,2); ndall = (1..5)).

Problems: 27 chebfun2-suite cases (reference = `tests/Polished_results`, with the 6.1 duplicate handled the same way
as in the test). 2D constructed: `diag_wW` = {sin(W(x+y)+.3), x-2y+.1} (exact roots; degree about W+40 in both
variables of f); `diag2_wW` = {sin(W(x+y)+.3), sin(W(x-y)/8+.2)} (exact lattice roots; both functions high-degree);
`sincos_wW` = {sin(Wx)-y, cos(Wy)-x} (reference = dense roots after Newton polishing; they agree to 1e-15 with
the other methods). 5D (the tests have no 5D systems): `tri5` triangular polynomial system (exact),
`trig5_kK` = sin(k_i (A x)_i + c_i) with A = I + 0.15R (exact lattice roots), and `diag5_wW` = full 5D tensor
sin(W*sum x+.3) plus a linear chain (exact roots on a line). The largest 5D degree is 19–23 per axis: 5D tensors
of degree >= 25 (25^5 coefficients) are already expensive to build and to subdivide, so 5D rootfinding never
reaches degrees where NUFFT pays off.

## Results (median seconds; roots compared with the reference by 1-1 matching at 1e-6·box width)

| problem | dim | max deg | n_ref | dense (s) | fast (s) | auto_nd12_d32 (s) | auto_nd12_d64 (s) | auto_nd12_d256 (s) | auto_ndall_d16 (s) | xform share dense / fast | xform calls (dense) | roots vs ref | max err dense / worst other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cf1.4 | 2 | 1 | 1 | 0.00088 | 0.00453 | 0.000899 | 0.000884 | 0.000884 | 0.000883 | 1% / 81% | 8 | all 1/1 | 0.0e+00 / 3.9e-17 |
| cf1.5 | 2 | 1 | 1 | 0.00103 | 0.00515 | 0.00106 | 0.00104 | 0.00105 | 0.00104 | 1% / 80% | 8 | all 1/1 | 1.6e-16 / 1.6e-16 |
| cf2.2 | 2 | 2 | 2 | 0.00184 | 0.0169 | 0.00191 | 0.0019 | 0.00189 | 0.00187 | 3% / 88% | 31 | all 2/2 | 4.8e-16 / 4.1e-14 |
| cf6.3 | 2 | 2 | 4 | 0.00411 | 0.105 | 0.00426 | 0.00418 | 0.00416 | 0.00414 | 7% / 96% | 200 | all 4/4 | 3.1e-16 / 9.7e-15 |
| cf9.2 | 2 | 2 | 2 | 0.00335 | 0.0844 | 0.00349 | 0.00344 | 0.00343 | 0.00339 | 7% / 96% | 158 | all 2/2 | 4.1e-16 / 3.2e-15 |
| cf1.3 | 2 | 3 | 5 | 0.00517 | 0.138 | 0.0054 | 0.00515 | 0.00528 | 0.00544 | 9% / 96% | 270 | all 5/5 | 4.3e-15 / 2.5e-14 |
| cf6.1 | 2 | 3 | 6 | 0.00527 | 0.131 | 0.0056 | 0.00541 | 0.00537 | 0.00536 | 9% / 96% | 318 | ref 6; fast:5 (-1/+0) | 1.1e-08 / 9.5e-08 |
| cf6.2 | 2 | 3 | 6 | 0.0054 | 0.148 | 0.00568 | 0.00553 | 0.00552 | 0.00553 | 9% / 96% | 282 | all 6/6 | 4.0e-13 / 5.5e-12 |
| cf7.1 | 2 | 3 | 4 | 0.00521 | 0.145 | 0.00549 | 0.00529 | 0.00528 | 0.0053 | 9% / 96% | 276 | all 4/4 | 2.4e-15 / 4.7e-14 |
| cf1.1 | 2 | 6 | 4 | 0.0039 | 0.106 | 0.00407 | 0.00387 | 0.00368 | 0.00365 | 11% / 96% | 208 | all 4/4 | 9.0e-16 / 4.0e-15 |
| cf3.1 | 2 | 6 | 4 | 0.00744 | 0.199 | 0.00734 | 0.00756 | 0.00718 | 0.00723 | 12% / 96% | 404 | all 4/4 | 9.9e-13 / 1.8e-10 |
| cf3.2 | 2 | 8 | 45 | 0.0905 | 2.44 | 0.093 | 0.0951 | 0.0938 | 0.0943 | 22% / 96% | 4872 | all 45/45 | 7.5e-12 / 7.5e-12 |
| cf7.2 | 2 | 10 | 10 | 0.0193 | 0.506 | 0.02 | 0.0198 | 0.0197 | 0.0197 | 16% / 96% | 956 | all 10/10 | 6.1e-09 / 1.1e-08 |
| cf1.2 | 2 | 12 | 13 | 0.0363 | 0.918 | 0.0345 | 0.0375 | 0.0365 | 0.037 | 25% / 96% | 1686 | all 13/13 | 7.2e-09 / 3.9e-08 |
| cf9.1 | 2 | 13 | 4 | 0.00429 | 0.0934 | 0.00449 | 0.00436 | 0.00436 | 0.00435 | 15% / 95% | 164 | all 4/4 | 1.6e-16 / 8.8e-16 |
| cf4.2 | 2 | 18 | 2 | 0.0107 | 0.161 | 0.0109 | 0.0107 | 0.0108 | 0.0226 | 17% / 93% | 304 | all 2/2 | 7.2e-13 / 2.2e-12 |
| cf7.3 | 2 | 19 | 2 | 0.00524 | 0.0818 | 0.0053 | 0.00519 | 0.00515 | 0.0754 | 26% / 95% | 138 | all 2/2 | 5.6e-25 / 9.5e-25 |
| cf4.1 | 2 | 20 | 5 | 0.00881 | 0.161 | 0.00913 | 0.00919 | 0.00901 | 0.148 | 31% / 95% | 288 | all 5/5 | 1.6e-16 / 1.1e-15 |
| cf2.3 | 2 | 23 | 5 | 0.00613 | 0.0995 | 0.00605 | 0.00592 | 0.00596 | 0.0687 | 25% / 94% | 164 | all 5/5 | 3.8e-16 / 8.7e-16 |
| cf8.1 | 2 | 33 | 8 | 0.00923 | 0.176 | 0.0335 | 0.00934 | 0.00928 | 0.116 | 25% / 95% | 304 | all 8/8 | 2.5e-16 / 1.5e-15 |
| cf2.1 | 2 | 34 | 6 | 0.00684 | 0.138 | 0.0627 | 0.00668 | 0.00733 | 0.0735 | 33% / 96% | 238 | all 6/6 | 2.5e-16 / 1.4e-15 |
| cf8.2 | 2 | 34 | 39 | 0.0306 | 0.769 | 0.304 | 0.0308 | 0.0306 | 0.327 | 26% / 96% | 1342 | all 39/39 | 5.7e-16 / 1.3e-15 |
| cf7.4 | 2 | 39 | 49 | 0.0351 | 0.867 | 0.302 | 0.0355 | 0.0354 | 0.512 | 26% / 95% | 1402 | all 49/49 | 9.4e-15 / 9.4e-15 |
| cf5.1 | 2 | 55 | 10 | 0.0333 | 0.587 | 0.394 | 0.033 | 0.0347 | 0.588 | 44% / 96% | 894 | all 10/10 | 9.0e-16 / 3.1e-15 |
| cf10.1 | 2 | 57 | 17 | 0.0146 | 0.292 | 0.0384 | 0.0148 | 0.0148 | 0.0762 | 20% / 95% | 512 | all 17/17 | 4.1e-14 / 4.3e-14 |
| cf2.5 | 2 | 61 | 103 | 0.138 | 3.17 | 1.22 | 0.144 | 0.143 | 3.01 | 37% / 96% | 4856 | all 103/103 | 4.0e-15 / 2.7e-14 |
| cf2.4 | 2 | 75 | 93 | 0.136 | FAIL (RecursionError) | 2.08 | 0.308 | 0.142 | 3 | 38% / - | 4524 | ref 93; fast:FAIL | 5.7e-16 / 2.3e-15 |
| sincos_w20 | 2 | 49 | 161 | 0.0914 | 2.65 | 1.07 | 0.0935 | 0.093 | 1.1 | 20% / 96% | 5326 | all 161/161 | 8.9e-16 / 2.6e-15 |
| sincos_w40 | 2 | 76 | 647 | 0.359 | 10.8 | 4.17 | 2.55 | 0.363 | 4.38 | 20% / 96% | 21004 | all 647/647 | 2.1e-15 / 5.0e-15 |
| diag_w50 | 2 | 89 | 47 | 0.0543 | 1.18 | 0.478 | 0.333 | 0.0544 | 0.61 | 32% / 95% | 1890 | all 47/47 | 6.0e-16 / 1.0e-15 |
| diag2_w100 | 2 | 147 | 1024 | 1.12 | 31.4 | 18.2 | 10.6 | 1.12 | 26.7 | 43% / 96% | 43112 | all 1024/1024 | 5.7e-16 / 1.1e-15 |
| diag_w100 | 2 | 147 | 96 | 0.122 | 2.64 | 0.913 | 0.735 | 0.124 | 1.09 | 39% / 95% | 3882 | all 96/96 | 4.7e-16 / 1.1e-15 |
| sincos_w100 | 2 | 148 | 4053 | 2.33 | 75.9 | 26.8 | 25.4 | 2.35 | 26.5 | 20% / 95% | 138120 | all 4053/4053 | 1.4e-15 / 4.0e-15 |
| diag2_w200 | 2 | 258 | 4033 | 4.91 | FAIL (timeout>120s) | 66.4 | 48.5 | 19.7 | 106 | 44% / - | 170000 | ref 4033; fast:FAIL | 5.6e-16 / 1.0e-15 |
| diag_w200 | 2 | 258 | 191 | 0.292 | 5.57 | 1.92 | 1.74 | 0.779 | 2.24 | 51% / 95% | 7704 | all 191/191 | 4.7e-16 / 1.1e-15 |
| diag_w400 | 2 | 473 | 382 | 0.935 | 13.1 | 5.88 | 5.69 | 6.04 | 6.74 | 66% / 95% | 15328 | all 382/382 | 5.8e-16 / 9.0e-16 |
| diag_w800 | 2 | 889 | 764 | 3.68 | 35.4 | 18.8 | 18.9 | 19.2 | 20.3 | 83% / 96% | 30568 | all 764/764 | 8.5e-16 / 1.2e-15 |
| tri5 | 5 | 2 | 2 | 0.00811 | 0.0454 | 0.00841 | 0.00821 | 0.00817 | 0.0082 | 2% / 78% | 66 | all 2/2 | 0.0e+00 / 9.1e-16 |
| trig5_k2 | 5 | 19 | 2 | 2.9 | 46.9 | 2.84 | 2.89 | 2.89 | 5.85 | 93% / 99% | 1450 | all 2/2 | 2.0e-15 / 2.8e-15 |
| diag5_w3 | 5 | 20 | 3 | 4.95 | 91.9 | 4.97 | 4.95 | 4.96 | 55.1 | 95% / 99% | 732 | all 3/3 | 8.5e-16 / 2.9e-15 |
| trig5_k3 | 5 | 22 | 29 | 44.1 | 856 | 43.7 | 43.7 | 45.7 | 73.9 | 97% / 100% | 16925 | all 29/29 | 1.7e-15 / 2.3e-15 |
| diag5_w4 | 5 | 23 | 5 | 10.7 | 124 | 10.6 | 10.7 | 10.6 | 81.2 | 95% / 99% | 812 | all 5/5 | 1.4e-15 / 2.0e-15 |

Extra run (`results_big_trunc_big.csv`): diag_w1600 (degree 1712): dense 18.3 s, auto_nd12_d256 57.3 s,
autotrunc_nd12_d64 **7.34 s**, fasttrunc 33.6 s; all find 1528/1528 roots.

Geometric-mean time relative to dense (`e5_summary.txt`):

```
2D-chebfun2     fast            time/dense geomean= 19.36 best= 4.98 worst=  27.8 failures=1/27
2D-chebfun2     auto_nd12_d32   time/dense geomean=  1.86 best= 0.95 worst=  15.3 failures=0/27
2D-chebfun2     auto_nd12_d64   time/dense geomean=  1.04 best= 0.96 worst=   2.3 failures=0/27
2D-chebfun2     auto_nd12_d256  time/dense geomean=  1.01 best= 0.94 worst=   1.1 failures=0/27
2D-chebfun2     auto_ndall_d16  time/dense geomean=  2.98 best= 0.94 worst=  22.0 failures=0/27
2D-constructed  fast            time/dense geomean= 21.48 best= 9.61 worst=  32.6 failures=1/10
2D-constructed  auto_nd12_d32   time/dense geomean=  9.29 best= 5.11 worst=  16.2 failures=0/10
2D-constructed  auto_nd12_d64   time/dense geomean=  5.90 best= 1.02 worst=  10.9 failures=0/10
2D-constructed  auto_nd12_d256  time/dense geomean=  1.81 best= 1.00 worst=   6.5 failures=0/10
2D-constructed  auto_ndall_d16  time/dense geomean= 11.02 best= 5.51 worst=  23.8 failures=0/10
5D              fast            time/dense geomean= 13.05 best= 5.60 worst=  19.4 failures=0/5
5D              auto_nd12_d32   time/dense geomean=  1.00 best= 0.98 worst=   1.0 failures=0/5
5D              auto_nd12_d64   time/dense geomean=  1.00 best= 0.99 worst=   1.0 failures=0/5
5D              auto_nd12_d256  time/dense geomean=  1.01 best= 0.99 worst=   1.0 failures=0/5
5D              auto_ndall_d16  time/dense geomean=  3.11 best= 1.01 worst=  11.1 failures=0/5
```

## Why the fast transform loses: the dense path truncates its output

`TransformChebInPlace1D` returns `transformedCoeffs[:maxRow]`. It stops growing the output degree once the
recurrence terms fall below 1e-32, so when a subinterval is small (alpha small) its output degree is far below
n and its cost is about n·m, not n². `FastTransform.cheb_affine_fast_axis0` always returns all n+1 coefficients
(the tail is ~1e-15 of rounding noise). Measured on a 474x474 tensor:
alpha=0.5 → dense returns 367 rows in 76 ms, NUFFT returns 474 rows in 21 ms; alpha=0.01 → dense returns 25 rows in
7 ms, NUFFT returns 474 rows in 16 ms. So even when a single NUFFT call is faster, the tensors it hands back do not
shrink as the solver zooms in, and every later operation (further transforms, checks, subdivisions) runs at full
size. The per-call cost data (`e5_percolumn_vs_degree.png`) also show a fixed NUFFT cost of about 10–200 us per
column at low degree (plan setup), while the dense path costs 0.05–1 us. In the subdivision tree, 99% of 2D
transform calls have degree <= 32 and all 5D calls have degree <= 23 (`e5_degree_distribution.png`).

**Truncated-NUFFT variant (experimental, emulated in the harness, not in yroots).** The NUFFT output is cut to the
same `maxRow` the dense recurrence would keep (`e5_table_trunc.md`). With this, `autotrunc_nd12_d64` beats dense
on the highest-degree problems: diag_w800 2.84 s vs 3.10 s (0.92x), and diag_w1600 7.34 s vs 18.3 s (**2.5x**).
It still loses elsewhere: diag_w200 is 2.3x slower, diag_w400 1.6x, diag2_w100 4.3x, diag2_w200 3.2x and
sincos_w40 1.8x. In 5D it is identical to dense. Truncating also makes chebfun2 2.4 solve correctly with
`fasttrunc`.

## Accuracy and robustness

* **Roots found:** whenever a solve finished, every method found the same roots as the reference, with one
  exception: **`fast` on chebfun2 6.1 found 5 of 6 roots.** It merges the two branches of the double root at the
  origin, which dense resolves as a pair. `fasttrunc` does the same.
* **Failures:**
  * **`fast` on chebfun2 2.4 fails with a `RecursionError`.** The final-step subdivision recurses without bound.
    Raising the recursion limit only lets it go deeper: depth 493, then 2493, then 9993 at a limit of 20000
    (99 s). Dense reaches depth 8 (`log_recursion_check_cf24.txt`). This is a genuine non-termination, probably
    because the untruncated ~1e-15 coefficient tail keeps the solver from shrinking or discarding boxes.
  * **`fast` on diag2_w200 exceeded the 120 s limit** (dense takes 4.9 s).
  * No `auto` configuration failed.
* **Root error:** the NUFFT paths are slightly less accurate. Well-conditioned problems stay at about 1e-15 either
  way, but ill-conditioned ones get worse:

  | case | dense error | fast error |
  |---|---|---|
  | 3.1 | 9.9e-13 | 1.8e-10 (fails the test's 1e-10 tolerance) |
  | 1.2 | 7.2e-9 | 3.9e-8 |
  | 6.2 | 4e-13 | 5.5e-12 |
  | 7.2 | 6.1e-9 | 1.1e-8 |

  Residuals rise by roughly 10x. `auto` settings that never reach the NUFFT path give results bit-identical to
  dense.

## Conclusions

* **2D:** the fast transform, as implemented in yroots (no output truncation), never makes the whole solve
  faster.
  * `fast` is 5–33x slower than dense and is also less robust: one non-terminating case, one timeout and one
    lost double root.
  * `auto` is only harmless when it almost never switches to NUFFT: nd12_d256 costs about 1.0x on the chebfun2
    suite. On high-degree problems it still loses, e.g. 2.7x on diag_w200, 6.5x on diag_w400 and 5.2x on
    diag_w800. That is because the untruncated output from a few top-level calls makes the whole subtree more
    expensive.
  * Lower thresholds (d32, d64, ndall_d16) are 2–24x slower.
  * The transform's share of solve time is 1–44% on the suite and 20–83% on the high-degree problems, so the
    transform is worth optimising. But subdivision keeps most calls at low degree, where the dense path wins by
    orders of magnitude.
  * The only speedup seen anywhere came from the **truncated** variant at degree >= ~900 (2.5x at degree 1712).
    Shipping it would require (a) truncating the NUFFT output to the dense `maxRow` (or an equivalent tail
    check) and (b) a threshold of about 64–256.
* **5D:** it never helps and should stay off.
  * Degrees stay at 23 or below, and the transform takes 93–97% of solve time.
  * The NUFFT path costs about 2–3 us per column against 0.05–1 us for dense, so `fast` is 5.6–19x slower
    (856 s vs 44 s on trig5_k3) and `auto_ndall_d16` is 1–11x slower.
  * The default `FAST_TRANSFORM_NDIMS=(1,)` is correct for 5D. The 5D bottleneck needs a faster dense
    (batched, small-n) kernel, not an O(n log n) one.

## Files
* `run_e5.py`: the benchmark harness (instrumentation, timeouts, truncated variant). `problems.py`: problem
  definitions. `plot_e5.py`: builds the tables and plots. `recursion_check_cf24.py`: the 2.4 recursion-depth check.
* Raw data: `results_{cf,c2,5d_part1,5d_part2}.csv` and `calls_*.csv` (per-call aggregates by ndim, degree and
  path). Merged into `e5_results_all.csv` and `e5_calls_all.csv`. Truncated variant: `results_trunc_variant_raw.csv`,
  `results_big_trunc_big.csv`, merged into `e5_results_trunc_variant.csv`.
* Tables: `e5_table.md`, `e5_table_trunc.md`, `e5_summary.txt`. Plots: `e5_time_ratio.png`,
  `e5_percolumn_vs_degree.png`, `e5_degree_distribution.png`. Logs: `log_*.txt`. `log_cf_c2_crashed.txt` is the
  first run, which crashed on the 2.4 RecursionError and led to the failure handling.
