# E3: N-d transforms, dense vs fast (NUFFT), in 2D and 5D

Branch `fast-transform`. Compared `TransformChebInPlaceND` (one axis) and `transformCheb` (all axes) with
`TRANSFORM_METHOD='dense'` (numba `TransformChebInPlace1D`) against `'fast'`
(`FastTransform.cheb_affine_fast_axis0`: one FINUFFT type-3 call with n_trans, then DCT-I).

## Setup

* Machine: 12-core arm64 Mac, shared with other benchmark agents (load average about 7 throughout).
  Everything ran single-threaded (`NUFFT_NTHREADS=1`, numba serial, `OPENBLAS/MKL_NUM_THREADS=1`).
  A `fast_mt4` variant (FINUFFT `nthreads=4`) was also timed. It was skipped for 5D tensors above
  2e5 entries because it was already 5 to 8 times slower than 1 thread there.
* Timing: one warm-up call (so numba JIT and the FINUFFT first call are excluded), then repeats until
  at least 0.3 s had passed (at least 7 reps, 3 for the largest cases, at most 400). The CSV gives the median plus the
  10th and 90th percentiles. The spread is small: p90/p10 is under 1.1 for 80% of rows and under 1.2 for 88% (max 3.1, from a few
  small-size outliers under load). The p10 to p90 bands are shaded in the plots.
  Times include the transposes and, on the fast path, its contiguous copies
  (`reshape(n+1,-1).T` then `ascontiguousarray`, and the back-transpose).
* Shapes: 2D squares n = 4 to 1024 (16 sizes), plus skewed shapes (1024,8), (8,1024), (1024,32),
  (32,1024), (256,16), (16,256), (128,8) and (8,128). 5D cubes n = 3 to 20 (11 sizes, up to 20^5 = 3.2M
  entries), plus skewed shapes (20,3,3,3,3), (3,3,3,3,20), (40,4,4,4,4), (64,3,3,3,3), (12,8,5,3,3) and (20,20,5,5,5).
  "n" means the per-axis size (degree n-1).
* Transforms (alpha, beta): subdivision `sub_lo`=(0.5,-0.5) and `sub_hi`=(0.5,0.5); zooms
  `zoom0.1`=(0.1,0.3) and `zoom1e-3`=(1e-3,0.2). The same (alpha, beta) is used on every axis.
* Coefficients: `decay` is N(0,1) times a geometric decay in every axis that reaches 1e-16 at the last
  index (a converged approximation). `unit` is N(0,1) with no decay.
* Accuracy reference: the exact transformation matrix, built with integer fixed-point arithmetic
  (2^-300 resolution) and rounded to double-double. It is applied with a compensated Dot2 (TwoProd plus TwoSum)
  matrix product. I checked it against mpmath (200-bit), and the difference was at most 2e-33. So the
  reference is accurate to about ulp of the result. The dense ErrorFree path (`exact=True`) is also reported
  as a method. The full-tensor reference applies the per-axis reference to axes 0, 1, ... in turn.
  Dense results that are row-truncated are zero-padded before comparison.

## Important context: the dense transform is not O(n^2) per fiber

`TransformChebInPlace1D` stops adding rows once the leading entry of a column drops below 1e-16. It then
returns only `maxRow` rows. For subdivision (alpha=0.5) maxRow grows sublinearly: 62 rows of 64, 113 of 128,
210 of 256, 398 of 512, 770 of 1024. For zooms it is tiny: alpha=1e-3 gives 6 to 14 rows for n = 8 to 1024. So the
dense cost is O(n * maxRow) per fiber, and dense is very cheap for zooms. The truncation is legitimate: the
dropped true coefficients are below 4e-15, and the accuracy runs confirm this. The fast path always returns all n rows,
so after a fast transform the tensor also stays larger for the next axes.

## Timing results

### Crossover (per-axis n where fast becomes faster; the ratio is dense time / fast time)

| | subdivision (0.5, +-0.5) | zoom alpha=0.1 | zoom alpha=1e-3 |
|---|---|---|---|
| 2D axis 0 | **128** (ratio 1.00; 2.5 at 256, 8.4 at 1024) | 384 (2.2 at 1024) | never (0.30 at 1024) |
| 2D axis 1 | 128 (1.4; 11 at 256, 64 at 1024) | 256 (16.7 at 1024) | 512 (2.2 at 1024) |
| 2D full `transformCheb` | **128** (1.04; 5.7 at 256, 28 at 1024) | 512 (2.4 at 1024) | never (0.16 at 1024) |
| 5D any axis, full | **never** (best 0.30 at n=20) | never (0.22 to 0.27) | never (0.04 to 0.15) |
| 5D skewed, long axis 0 | (64,3,3,3,3) axis 0: 0.95; (40,4,4,4,4): 0.60 | | |

Decay and unit coefficients give the same timings to within a few percent, because neither path depends
on the values.

2D axis 1 favors fast more strongly than axis 0 for a dense-side reason. `TransformChebInPlace1D` on the
transposed view walks strided rows. At n >= 256 it becomes cache-bound: at n=1024, dense axis 1 takes
5.6 s against 0.69 s for axis 0. The fast path copies to contiguous memory, so both of its axes cost the same (83 ms).
Much of the large 2D full-tensor speedup at n >= 256 comes from this dense inefficiency. A contiguous copy
before the dense call would reduce it.

### Realistic yroots sizes

2D, full `transformCheb`, subdivision `sub_hi`, unit coefficients (median ms, p10 to p90 in parentheses):

| n | dense | fast | dense/fast |
|---|---|---|---|
| 8 | 0.010 | 0.93 | 0.011 (fast 92x slower) |
| 16 | 0.028 | 0.98 | 0.028 (35x slower) |
| 32 | 0.12 | 1.15 | 0.11 (9.5x slower) |
| 64 | 0.69 | 1.76 | 0.39 (2.5x slower) |
| 96 | 1.93 | 2.52 | 0.76 (1.3x slower) |
| 128 | 4.41 (4.40 to 4.44) | 4.26 (4.21 to 4.32) | 1.04 (break-even) |
| 256 | 63.1 | 11.1 | 5.7 |
| 1024 | 4936 | 163 | 30 |

At alpha=1e-3, fast is 5 to 100 times slower at every size up to 1024. For example, at n=64 dense takes 0.11 ms and fast 1.64 ms.

5D, full `transformCheb`, `sub_hi`, unit coefficients:

| n | dense ms | fast ms | fast slowdown |
|---|---|---|---|
| 3 | 0.025 | 2.87 | 115x |
| 4 | 0.067 | 4.56 | 68x |
| 5 | 0.22 | 8.33 | 39x |
| 6 | 0.58 | 14.8 | 26x |
| 8 | 2.99 | 47.2 | 16x |
| 10 | 10.9 | 115 | 10.5x |
| 12 | 32.0 | 246 | 7.7x |
| 16 | 170 | 849 | 5.0x |
| 20 | 704 | 2362 | 3.4x |

Extrapolating the ratio, which grows about like n^1.8, puts a 5D crossover near n of roughly 35 to 40 per axis, which is 50M to 100M entries.
That is far beyond the sizes yroots uses. For zoom alpha=1e-3 in 5D, fast is 22 to 116 times slower at all sizes. Skewed 5D shapes
are worse still, because the short axes cost the fast path about 0.6 ms or more each (ratios 0.01 to 0.03).

### Why fast loses in 5D (and in small 2D)

See `breakdown.py` and `breakdown.txt`. The FINUFFT type-3 call takes about 97% of the fast path's time.
It has a fixed cost of about 0.45 ms per call, then about 2.2 to 2.8 us per 1D transform (n_trans) in 5D, even for
n=3 to 20. Dense costs about 0.2 us per fiber at n=10. The DCT and the copies are only 5 to 15%. More FINUFFT
threads do not help: for the full 2D tensor at n >= 512, `fast_mt4` took 1.0 to 1.9 times as long as 1 thread,
and it never helped by more than a few percent on any single axis. At small n it was 1.5 to 5 times slower,
and in 5D it was 5 to 8 times slower (on a loaded machine).

## Accuracy results (max abs error / sum|M|, compared with the bound n*2^-52*sum|M|)

Accuracy was checked on all shapes above (2D up to 1024^2, 5D up to 20^5), with 4 transforms, 2 coefficient types,
every axis, and the full tensor. That is 4,176 comparisons.

| ndim | method | coeffs | max err/bound | median err/bound | max err/sum abs(M) | violations |
|---|---|---|---|---|---|---|
| 2 | dense | decay | 0.15 | 2.2e-3 | 4.3e-16 | 0/288 |
| 2 | dense | unit | 0.020 | 1.0e-4 | 9.3e-17 | 0/288 |
| 2 | exact | both | 0.15 | 2e-3 / 7e-5 | 4.3e-16 | 0/576 |
| 2 | fast | decay | **0.82** | 8.9e-3 | 3.3e-15 | 0/288 |
| 2 | fast | unit | 0.46 | 8e-4 | 5.1e-16 | 0/288 |
| 5 | dense | decay | 0.22 | 8.7e-3 | 5.7e-16 | 0/408 |
| 5 | dense | unit | 0.007 | 1e-4 | 9.0e-18 | 0/408 |
| 5 | exact | both | 0.22 | | 5.7e-16 | 0/816 |
| 5 | fast | decay | **1.27** | 0.28 | 3.9e-15 | **9/408** |
| 5 | fast | unit | 0.077 | 3e-4 | 1.1e-16 | 0/408 |

(Each row counts the per-axis and full-tensor rows together.)

* **Bound violations: 9, all from the fast method on 5D 3x3x3x3x3 with decaying coefficients** (`sub_lo` on
  all 5 axes; `sub_hi` on axes 0, 2, 3 and 4). err/bound is 1.04 to 1.27, and err/sum|M| is about 7e-16 to 8.4e-16,
  against a bound of 3*2^-52 = 6.7e-16. Full-tensor errors stayed within the accumulated bound (at most 0.78 of it).
  Near-misses in the same regime: 5D n=4 and n=5 decay at 0.45 to 0.90 of the bound, and 2D 4x4 decay at 0.82.
* Cause: FINUFFT's error is relative to the l1 norm of each fiber's strengths, which gives an absolute floor of about
  (5 to 8)e-16 times |a_0| that does not depend on n. With decaying coefficients, sum|M| is about the size of the leading
  entry, so the fast error is about 3e-16 to 1e-15 times sum|M| at every n. The bound n*2^-52 shrinks with n and drops below that floor
  at n=3. The fast path's error does not grow with n, so it is comfortably inside the bound for n >= 8.
* In the same cases the fast error is typically 3 to 40 times the dense error. Dense and exact (ErrorFree) are
  essentially the same, both limited by the final rounding of O(1) outputs, and both stay at or below 0.22 of the bound.

## Conclusions

1. **2D crossover: per-axis n of about 128 for subdivision** (both per axis and for the full tensor). It is about 384 to 512
   for alpha=0.1 zooms, and there is none for alpha=1e-3 zooms, where dense row truncation makes dense O(n*10).
   At realistic 2D sizes (n = 10 to 100) the fast method is 1.3 to 50 times slower.
2. **5D: the fast method never wins** at any tested size up to 20^5. It is 10 to 115 times slower at realistic sizes (n = 3 to 10)
   and 3.4 times slower at 20^5. The fixed 0.45 ms FINUFFT call overhead and about 2.5 us per fiber dominate.
   The only case close to break-even is a single long axis, (64,3,3,3,3) along axis 0 at 0.95.
3. The current `auto` defaults (`FAST_TRANSFORM_NDIMS=(1,)`, `MIN_DEGREE=256`) are safe. If 2D is ever enabled,
   the threshold should be about 256 (a clear 2.5 to 5.7x win), applied only when alpha >= about 0.1, because the
   crossover depends strongly on alpha through dense truncation. It should never be enabled for 5D.
4. Using the fast method at small n (<= 5 per axis) with converged coefficients can exceed the
   solver's error bound by up to 1.27x. `auto` would never choose fast there, but `TRANSFORM_METHOD='fast'` does.

## Files

* `common.py`: coefficient generator, exact/double-double reference matrix plus Dot2 apply, and timing helper.
* `run_timing.py` writes `timing.csv` (one row per shape/param/coeffs/op/method, with median/p10/p90/reps and dense output shape).
* `run_accuracy.py` writes `accuracy.csv` (errors, bound, violation flag).
* `analyze.py` writes `timing_2d.png`, `timing_5d.png`, `speedup.png`, `accuracy.png`, `crossover.csv`,
  `skewed_summary.csv` and `analysis_output.txt`.
* `breakdown.py` writes `breakdown.txt` (the fast path split into its NUFFT, DCT and copy costs).
* `cache/`: reference matrices, cached by `common.py` (134 MB; deleted after the run and rebuilt automatically on rerun).

Rerun with (from the repo root):
`uv run --no-sync python experiments/fast_transform/e3_nd_transform/run_timing.py 2 5`,
then the same for `run_accuracy.py` and `analyze.py`.
