# NUFFT affine Chebyshev transform in yroots: summary of experiments E1–E5

Branch `fast-transform` (commit b0f3f1d adds `yroots/FastTransform.py` and the
`TRANSFORM_METHOD` switch; default `'dense'`). Each subdirectory has its own `results.md`.
All timings single-threaded, medians, on a shared 12-core machine (±15–20% noise end-to-end).

## Verdict
As implemented, the NUFFT transform never speeds up an end-to-end solve in dim 1, 2 or 5.
With output truncation added (a prototype only, not in yroots), it gives a modest win in 1D at
degree ≳ 4000, and in 2D at degree ≳ 900.

## Single transform (E1, E3)
- Fixed cost ≈ 450 µs per call, ~95% of it FINUFFT type-3 setup, which runs on every call
  because the target nodes change with (α, β, n). Dense takes ≈ 1 µs at small n.
- 1D crossover: n ≈ 1200 for half-splits (18× faster at 1e4, 130× at 65535); n ≈ 3500 for α = 0.1;
  never for α ≤ 1e-3.
- The dense routine truncates its output: after mapping to a subinterval of width α it keeps
  O(n·α + log) rows. That makes zoom steps cheap, and dense stays faster there.
- 2D crossover ≈ 128 per axis (subdivision). 5D: no crossover up to 20⁵; 10–115× slower at 3–10 per axis.
- Side finding: dense on a non-leading axis reads a strided transposed view. At 1024²,
  axis 1 takes 5.6 s against 0.69 s for axis 0. A contiguous copy should close most of that gap.

## Accuracy (E2, E3)
- Fast (eps = 1e-15) is 10–40× less accurate than dense per transform. Asking for eps below 1e-14
  gains nothing for n ≥ 256.
- `getTransformationError` = n·2⁻⁵²·‖M‖₁ is exceeded by fast in 9% of 1D cases (up to 22×) and on
  5D 3⁵ tensors (1.27×). Dense exceeds it too, up to 3.6× for random coefficients at n ≥ 512,
  so the bound is not rigorous even now. Only the ErrorFree variant never exceeded it.
- The main error source in the split cases is `arccos(αcosθ+β)` near ±1, which is ill-conditioned.
  Computing φ = 2·arcsin(√((1−y)/2)), with 1−y formed without cancellation, brings the worst cases
  to about 0.4× the bound.
- `canUseFast` accepts (0.1, −0.9) because α+|β| rounds to 1, although the exact map leaves [−1, 1].
- Fast leaves ~1e-16·‖a‖₁ noise in every coefficient, so `trimMs` cannot shorten the result.
  Over repeated transforms the accumulated error stays under the accumulated bound.

## End-to-end (E4: 1D, E5: 2D/5D)
- 1D: every problem (degree 3–20245) is slower with fast or auto at every threshold from 64 to 4096. The
  untrimmed n+1 output makes the children keep full degree: for sin(1000x), calls at degree ≥ 256 go from 18 to 602.
  Transform share of dense solve time: 2–5% below degree 1600, 23% at degree 20245. That caps the gain.
- 1D with truncation prototype (auto ≥ 1024): about even up to degree 3000; then −7 to −26%
  (sin 20000x: 6.77 → 5.00 s).
- 2D: `fast` is 5–33× slower and less robust:
  - cf2.4 hits RecursionError
  - diag2_w200 times out (dense: 4.9 s)
  - cf6.1 loses one root of a double root
  - cf3.1 error goes from 1e-12 to 2e-10
  - with truncation, speedup appears only at degree ≥ 900 (2.5× at degree 1712)
- 5D: the transform is 93–97% of solve time, but degrees are ≤ 23. Fast is 5.6–19× slower.
  In 5D the need is a faster dense kernel.
- Test suite with fast forced: 12 failures, all in 2D. Dense and auto (1D only): all pass.

## If pursued further
1. Truncate the fast output like dense does, and add the dropped ℓ¹ mass to the error.
2. Use the stable φ formula, and a strict domain check for `canUseFast`.
3. Gate on degree and on α: 1D n ≥ ~1024–4000 and α ≳ 0.1; 2D only at very high degree; never 5D.
4. Replace the error bound for the fast path with one that includes the NUFFT tolerance and a
   term that does not shrink with n. Revisit the dense bound too.
5. Independently: make the dense transform read a contiguous array on non-leading axes.
