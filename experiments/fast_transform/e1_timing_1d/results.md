# E1: 1D transform timing, dense vs fast

Branch `fast-transform`. Apple M2 Max (12 cores), Python 3.14t, numba 0.65.1, finufft 2.5.1, scipy 1.18.1.
Other benchmarks were running on the machine at the same time (load average 4 to 8).
Everything ran on one thread (`NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`, FINUFFT `nthreads=1`) unless stated otherwise.
Each point is the median of 5 to 2000 repeats (about 0.25 s per point, 3 repeats when one call takes more than 1 s), after one warmup call. Numba compile time is excluded.

- **Dense**: `TransformChebInPlace1D`.
- **Fast**: `cheb_affine_fast_axis0(nthreads=1)` with eps=1e-15.
- **nd_dense / nd_fast**: `TransformChebInPlaceND(c, 0, a, b, False)` with `TRANSFORM_METHOD` set to 'dense' or 'fast'.
- **Grid**: N = n+1 in {2, 3, 4, 6, 8, 12, ..., 49152, 65536}.
- **Coefficients**: `unit` is U(-1,1). `decay` is U(-1,1)·10^(-16k/n).

Files:
- `bench_1d.py`: the main sweep. Writes `timings.csv` (raw), `accuracy.csv`, `breakdown.csv` and `eps_threads.csv`.
- `analyze.py`: writes `time_vs_degree.png` and `summary_tables.md` (the crossover table plus median times at selected degrees). Its console output is in `analyze_out.txt`.
- `threads_rerun.py`: writes `threads_rerun.csv` and `threads_rerun.txt`. This is the thread comparison run without the OpenMP cap (see anomaly 1).
- `finufft_overhead_probe.py`: writes `finufft_overhead_probe.txt`, FINUFFT's internal debug timings for a small type-3 call.

## Crossover degrees and speedups (speedup = dense time / fast time, direct calls)

| case | alpha, beta | crossover n (direct / via ND) | n=1e3 | n=1e4 | n=65535 |
|---|---|---|---|---|---|
| half_left | 0.5, -0.5 | 1180-1200 / 1180-1190 | 0.75-0.78x | 18.2-18.6x | 131-132x |
| half_right | 0.5, 0.5 | 1180-1190 / 1180 | 0.77x | 18.2-18.3x | 131-133x |
| offcenter | 0.501, -0.499 | 1170-1190 / 1180 | 0.76-0.78x | 18.4-19.5x | 137x |
| zoom_0.1 | 0.1, 0.3 | ~3470 / 3480-3590 | 0.19-0.20x | 3.3-3.4x | 24x (decay value 67x was a load spike) |
| zoom_1e-3 | 1e-3, -0.2 | none up to 65535 | 0.025x | 0.13x | 0.45x |
| zoom_1e-6 | 1e-6, 0.7 | none up to 65535 | 0.016x | 0.042x | 0.052x |

Where a cell gives a range, the range covers both coefficient types. The full table, split by coefficient type, is in `summary_tables.md`.

- The crossover sits between the grid points n=1023 and n=1535 for the halving cases, and between n=3071 and n=4095 for alpha=0.1. The exact values are interpolated on a log-log scale.
- The current default `FAST_TRANSFORM_MIN_DEGREE = 256` in 'auto' mode is **too low**. At n=255 the fast path is about 14x slower (dense 38 µs against 519 µs for fast). A threshold of about 1200 would fit the halving transforms.
- The transform being applied matters more than the degree. For small alpha the dense code is nearly linear in n (see below), so the fast path never wins when alpha ≤ 1e-3. For alpha=1e-3 the ratio rises from 0.025x to 0.13x to 0.45x across n=1e3, 1e4 and 65535. Extrapolating, the crossover would be somewhere above n≈1e5. For alpha=1e-6 the fast path is always about 20x slower at large n.
- The ND dispatch overhead is negligible: 0.13 µs extra for dense and about 0.3 µs extra for fast (within noise). The crossovers through `TransformChebInPlaceND` match the direct ones.

## Dense cost depends on alpha, not on coefficient decay

The `1e-16` truncation in `TransformChebInPlace1D` tests `finalVal = alpha*arr2[i]`, and that value depends only on alpha and beta, not on the coefficients. So `decay` and `unit` coefficients have the same dense cost (the curves overlap in the plot), and the returned length is the same for both.

Dense output length at n=65535 by alpha:

| alpha | dense output length |
|---|---|
| ≈0.5 | 46505-46552 |
| 0.1 | 7056 |
| 1e-3 | 107 |
| 1e-6 | 7 |

The fast path always returns all n+1 coefficients. The entries past the dense truncation point come out at noise level: at most 6e-13 for decay and 3e-10 for unit.

Dense time at n=65535:

| alpha | dense time at n=65535 |
|---|---|
| ≈0.5 | 2.06 s |
| 0.1 | 0.31 s |
| 1e-3 | 5.6 ms |
| 1e-6 | 0.6 ms |

At small n the dense cost is 0.58 µs at n=1, 0.63 µs at n=7 and 0.8-1.3 µs at n=31.

## Fast-path fixed overhead (n ≤ 32, alpha=0.5)

The whole call takes **about 447 µs** (median; 446-452 µs for n=1-31). The breakdown by stage, measured with an instrumented copy of the function (`breakdown.csv`):

| stage | µs |
|---|---|
| reshape/transpose | 0.4 |
| nodes (cos/arccos/clip) | 6.0 |
| complex coefficient setup | 3.5-3.9 |
| **`finufft.nufft1d3`** | **427** |
| DCT-I and scaling | 9 (the bare scipy `dct` call is 3.3) |
| output copy | 0.4 |

Inside the NUFFT, the `Plan()` constructor takes 7 µs, `setpts` about 405 µs and `execute` about 10 µs. With a reused plan, `execute` alone is 9.5 µs.

FINUFFT's debug output shows that `setpts` for type 3 does three expensive things on every call:
- recomputes the kernel Horner coefficients (about 157 µs at eps=1e-15, ns=16);
- computes the phase and deconvolution factors (about 190 µs);
- builds and sets up the inner type-2 plan (about 210 µs).

These timings come from the debug run and overlap somewhat.

A type-3 plan cannot be reused, because the target points phi depend on alpha, beta and n. The cost is mostly independent of n and does depend on eps: 428, 280, 182 and 171 µs at eps = 1e-15, 1e-12, 1e-9 and 1e-6. Type-1 and type-2 calls of the same size cost about 245-250 µs, so a lot of this is FINUFFT's per-plan cost. Growth with n only takes over after about n=1000: the fast call is 670 µs at N=1024, 1.27 ms at 4096, 3.99 ms at 16384 and 15 ms at 65536. At N=65536 the NUFFT takes 12.7 ms, the DCT 2.1 ms and the node computation 0.5 ms.

## NUFFT_EPS and threads (alpha=0.5, unit coefficients)

| N | eps 1e-15 | 1e-12 | 1e-9 | max err vs dense (1e-15 / 1e-12 / 1e-9) |
|---|---|---|---|---|
| 1024 | 0.68 ms | 0.55 ms | 0.38 ms | 6.6e-13 / 3.5e-12 / 2.8e-9 |
| 4096 | 1.27 ms | 1.04 ms | 0.85 ms | 2.1e-12 / 4.9e-12 / 4.1e-9 |
| 16384 | 3.99 ms | 3.51 ms | 3.04 ms | 2.7e-11 / 2.7e-11 / 3.2e-9 |
| 65536 | 15.5 ms | 13.9 ms | 12.0-12.5 ms | (dense reference not computed) |

Relaxing eps to 1e-9 roughly halves the fixed overhead but gains only about 20% at N=65536. The error grows with it, to about 1e-9 times the coefficient scale.

Threads, measured without the OpenMP cap (`threads_rerun.txt`, eps=1e-15):

| N | nthreads=1 | 2 | 4 | 0 (auto, 12) |
|---|---|---|---|---|
| 1024 | 0.68 ms | 0.88 ms | 1.03 ms | 1.75 ms |
| 4096 | 1.36 ms | 1.47 ms | 1.32 ms | 2.18 ms |
| 16384 | 3.99 ms | 3.17 ms | 2.82 ms | 3.43 ms |
| 65536 | 15.5 ms | 10.6 ms | 8.2 ms | 8.7 ms |

Multithreading does not pay off below N≈8k. The auto setting (all 12 cores, on a loaded machine) is 2.6x slower at N=1024. At N=65536, 4 threads give 1.9x. Keeping `NUFFT_NTHREADS = 1` as the default is right, and yroots' own parallelism works across subintervals anyway.

## Accuracy check (from the timing sweep, eps=1e-15)

The largest |dense - fast| over the leading entries the two outputs share:

| coefficients | max difference | over |
|---|---|---|
| decay | 5.6e-12 | all cases and n |
| unit | 4.9e-10 | all cases and n |

The unit difference occurs at N=65536, where ||a||₁ ≈ 3e4, so relative to ||a||₁ it is about 1e-14. The dense code has its own O(n·eps) rounding, so this difference is not the error of the fast path alone.

## Anomalies and caveats

1. **The original `nthreads=0` measurement is invalid.** In `eps_threads.csv`, `nthreads=0` ran under `OMP_NUM_THREADS=1`, so "auto" meant 1 thread. Use `threads_rerun.csv` for thread effects.
2. **Two outliers are load spikes.** decay/zoom_0.1 at n=65535 (dense 0.87 s against 0.31 s for unit) and decay/offcenter at n=32767 (1.12 s against 0.51 s) did not reproduce. Re-timing gave 0.315 s and 0.517 s, identical to unit, with no subnormals in the outputs. They come from interference by the concurrent runs. The decay/zoom_0.1 speedup of 67x at n=65535 should therefore read about 24x.
3. **The two paths return different lengths.** Dense output length is capped by the alpha-based truncation; fast always returns n+1 coefficients. A downstream step that trims or checks convergence may therefore see a different array shape.
4. The run-to-run spread is small. p90/p10 has a median of 1.05 for dense and 1.07 for fast, with a maximum of about 1.65 for dense and 1.18 for fast.
5. FINUFFT warns that eps=1e-15 is too small at upsampfac=1.25. This comes from a probe only; the default upsampfac=2 chooses ns=16 without a warning.
