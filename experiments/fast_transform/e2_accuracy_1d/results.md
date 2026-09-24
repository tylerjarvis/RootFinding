# E2: accuracy of the 1D affine Chebyshev transform (dense vs NUFFT fast)

Branch `fast-transform`. Methods compared:

- `dense`: `TransformChebInPlace1D`
- `dense_ef`: `TransformChebInPlace1DErrorFree`
- `fast_<eps>`: `FastTransform.cheb_affine_fast_axis0` with FINUFFT eps set to 1e-15, 1e-14 or 1e-12, `nthreads=1`

Everything was run single-threaded with `uv run --no-sync python`.

## Reference (the exact result)

`ddref.py` runs the column recurrence C_k = 2βC_{k-1} + α S(C_{k-1}) − C_{k-2} in vectorized **double-double arithmetic**, about 106 bits. It never truncates, and it accumulates b = Σ a_k C_k in double-double as well. The inputs a, α and β are treated as exact doubles. For repeated transforms the reference carries a double-double coefficient vector through the path, so it is the exact composite transform. Two checks (`validate_ref.py`, `validate_ref.log`):

- For n ≤ 200 and 5 (α, β) pairs, the reference matches an **mpmath dps=60** recurrence to max error ≤ 1e-32·‖a‖₁.
- For n = 2048 and 8192, a separate pointwise check agrees to ≤ 7e-32·‖a‖₁. It compares q(x) against p(αx+β), both evaluated with double-double Clenshaw at a double-double argument, at 40 random points.

Reference error therefore sits 16 or more orders below every error reported here.

## Setup

**Single transform** (`run_single.py`, output `single_transform.csv`, 5,720 rows):

- Degrees: n = 8, 16, …, 8192.
- 13 coefficient profiles:
  - Smooth functions: sin(n/2·x), eˣcos 50x, and Runge 1/(1+25x²).
  - A polynomial with n/2 random roots in [−1,1], 3 seeds.
  - N(0,1) random coefficients, 3 seeds.
  - Geometric decay down to 1e-16 at k = n, 3 seeds.
  - Geometric decay with ratio 0.8.
- 8 transforms:
  - Splits (0.5, ±0.5).
  - yroots' off-centre split at m = 0.0394555…, both halves.
  - Interior zooms with α = 0.1, 1e-3, 1e-6.
  - An edge zoom (0.1, −0.9).

**Bound check.** The bound is `getTransformationError` = (n+1)·2⁻⁵²·‖a‖₁. The solver treats it as a sup-norm error bound, so I compare two quantities against it:

- ‖b − b_exact‖₁, a rigorous upper bound on the sup-norm error.
- The sup-norm error on [−1,1], measured on an 8n-point Chebyshev grid.

## 1. Single-transform error

All 1,144 cases per method, errors relative to ‖a‖₁:

| method | max-abs coeff err: median / worst | ‖err‖₁: median / worst | sup err: median / worst | ‖err‖₁/bound: median / worst | bound violated (‖·‖₁ / sup) |
|---|---|---|---|---|---|
| dense | 6.4e-17 / 3.0e-15 | 2.0e-16 / 6.6e-12 | 1.5e-16 / 6.6e-12 | 0.008 / **3.6** | 32 / 29 |
| dense_ef | 3.2e-17 / 5.9e-16 | 1.3e-16 / 3.3e-15 | 8.6e-17 / 1.3e-15 | 0.003 / 0.13 | **0 / 0** |
| fast 1e-15 | 5.1e-16 / 1.3e-13 | 7.0e-15 / 4.0e-11 | 2.8e-15 / 3.2e-11 | 0.27 / **22** | 180 / 104 |
| fast 1e-14 | 1.1e-15 / 1.2e-13 | 2.3e-14 / 4.0e-11 | 1.1e-14 / 3.2e-11 | 0.46 / 22 | 416 / 294 |
| fast 1e-12 | 4.1e-14 / 1.2e-12 | 9.2e-13 / 4.0e-11 | 3.4e-13 / 3.2e-11 | 9.9 / 1880 | 943 / 800 |

See `single_error_vs_n.png` and `single_sup_vs_bound.png`. More tables are in `summary_tables.txt`.

- **Typical accuracy.** Fast at eps = 1e-15 is about 10–30× less accurate than dense per coefficient, and about 35× in ‖·‖₁ at the median. Its ‖·‖₁ error grows faster than n. At n = 8192 the median relative ‖·‖₁ error is 3.6e-13 for fast, against 8.8e-16 for dense.
- **eps 1e-14 vs 1e-15.** For n ≥ 256 the two settings give identical errors, so FINUFFT's accuracy saturates and asking for 1e-15 gains nothing at large n. eps = 1e-12 breaks the bound almost everywhere, 82% of cases.
- **Where fast 1e-15 breaks the bound** (sup norm): 104 of 1,144 cases.
  - **Splits** (every yroots subdivision): 80 cases, growing with n (16 of 52 split cases at n = 8192).
  - **Edge zoom**: 21 cases.
  - **Interior zooms**: 3 cases, all at n ≤ 16.
  - By profile, the violations are mostly random O(1) coefficients (worst 22× bound at n = 8192, off-centre split) and the n/2-roots polynomial (up to about 4×). For n ≤ 16 they occur across all profiles by up to about 2×, because at small n the bound (n+1)u is below FINUFFT's attainable accuracy.
  - Smooth or decaying profiles (sin, Runge, geometric) at n ≥ 32 never break the bound.
- **Dense also breaks the bound.** It fails for random coefficients at n ≥ 512 on splits and the edge zoom, worst 3.6× at n = 8192. `getTransformationError` is not rigorous even for the existing code; the code comment already says "TODO: … more rigorous bound". Only `dense_ef` stays within the bound in every case.

### Root cause of fast's excess error on splits: arccos conditioning at ±1

`FastTransform` computes φ_j = arccos(α·cos θ_j + β). A split maps one end of [−1,1] onto ±1, where arccos has infinite derivative, so rounding in α·cos θ_j + β turns into φ errors of about √u:

| n | measured max \|Δφ\| |
|---|---|
| 512 | 5e-15 |
| 8192 | 3.3e-13 |

`diag_endpoint.py` / `diag_endpoint.csv` recompute the same NUFFT with correctly rounded φ_j (mpmath):

| case (random coefficients) | numpy φ: ‖err‖₁/bound | exact φ: ‖err‖₁/bound |
|---|---|---|
| split_R, n = 8192 | 8.5 | **0.40** |
| offsplit_R, n = 8192 | 22.2 | **0.39** |
| offsplit_R, n = 2048 | 1.2 | 0.33 |
| interior zoom 0.1, n = 8192 | 0.25 | 0.22 (unchanged) |
| edge zoom (0.1, −0.9), n = 8192 | 8.2 | 5.8 |

**Suggested fix:** compute φ near the endpoints from 1 ∓ y without cancellation. Use 1 − cos θ = 2 sin²(θ/2), and for a split α = β = ½ take φ = 2·arccos(cos(θ/2)) type identities. In general, φ = 2·arcsin(√((1−y)/2)) with 1 − y formed as α(1 − cos θ) + (1 − α − β). With that fix, splits drop to about 0.4× the bound.

The edge-zoom residual has a separate cause. In floats 0.1 + 0.9 == 1.0, so `canUseFast` accepts (0.1, −0.9). In exact arithmetic the map reaches −1 − 5.6e-17, just outside [−1,1], and φ is clipped there.

## 2. Repeated transforms (paths of depth 10 / 30 / 60)

`run_paths.py` produces `repeated_transform.csv` and `accumulated_error_vs_depth.png`. `run_paths_local.py` produces `repeated_local_steps.csv` and `local_step_error_vs_bound.png`.

**Setup:**

- Starting polynomials: the n/2-roots polynomial (2 seeds) and sin(n/2·x), at n = 64, 512 and 4096.
- Each path steers toward a known root. Three schedules:
  - **split**: 60 off-centre half-splits.
  - **mixed**: split and zoom alternating, zoom factor U[0.05, 0.5].
  - **zoomdeep**: zooms of 1e-3 / 0.1 / 0.1 repeating.
- Each method feeds its own output into the next step. Dense arrays shrink through its pruning; there is no `trimMs`.
- The accumulated bound is Σ_i getTransformationError(M_i).

**Accumulated error**, ‖b_d − b_exact,d‖₁ / ‖a₀‖₁, worst over 9 paths per schedule (3 starting polynomials × 3 degrees):

| depth | dense | dense_ef | fast 1e-15 | fast 1e-14 | fast 1e-12 |
|---|---|---|---|---|---|
| 10 | 5.7e-16 | 3.3e-16 | 5.2e-14 | 5.6e-14 | 1.1e-12 |
| 30 | 5.7e-16 | 3.2e-16 | 2.1e-14 | 2.2e-14 | 8.6e-14 |
| 60 | 5.7e-16 | 3.2e-16 | 2.1e-14 | 2.2e-14 | 8.6e-14 |

- **Error does not grow with depth.** Each later transform maps onto a subinterval, so it cannot increase the sup norm, and restricting noise to a subinterval shrinks its coefficient ℓ₁ norm. The accumulated error is set in the first few steps, while ‖M‖ is still O(1), and then stays flat.
- **Accumulated error vs accumulated bound**, worst case:

  | method | worst ratio | above 1? |
  |---|---|---|
  | dense | 0.004 | no |
  | fast 1e-15 | 0.03 | no, except at depth 1 |
  | fast 1e-14 | 0.06 | no, except at depth 1 |
  | fast 1e-12 | 4.4 | yes, at depth ≤ 10 |

  At depth 1 the accumulated bound is just the single-step bound, which fast 1e-15 breaks by 1.6–1.8× on the first split for n = 512 and 4096.
- **Per-step (local) check**: each step's error is measured against the exact transform of that method's own input.

  | method | worst per-step ratio | steps above bound |
  |---|---|---|
  | dense | 0.11 | 0 |
  | fast 1e-15 | 1.77 | 4 of 1,350 (all on the first split from the full-size polynomial) |
  | fast 1e-12 | 420 | 97% |

- **Relative to the current polynomial size** (not ‖a₀‖₁), errors look large for every method, dense included (median 0.1–0.6). This is expected: after zooming toward a root the polynomial itself is tiny, and the O(u‖a₀‖) error from the early steps dominates it. The solver's accumulated bound covers this, as the ratios above show.

## 3. Where the error sits in the coefficients; noise floor in the tail

See `error_along_index_n1024.png` and `error_along_index_n8192.png`.

**The fast error is spread flat across all n+1 coefficients (white noise).**

| method | median share of ‖err‖₁ in the top half (k > n/2) | median share in the top 10% |
|---|---|---|
| fast 1e-15 | 0.38–0.39 | 0.07–0.09 |
| dense | ≈ 0 | ≈ 0 |

For fast these shares are close to the uniform values of 0.5 and 0.1. For dense the shares are essentially zero because its error follows the true coefficient envelope, and the envelope is tiny in the tail after a zoom or for decaying profiles.

**Noise floor.** These coefficients have true value below 1e-20·‖a‖₁ (for example k ≥ 3 after a zoom of 1e-3); at n = 8192 there are about 5,500 of them. The values each method returns in those slots, relative to ‖a‖₁:

| method | per coefficient: median / worst | total over those slots at n = 8192 (interior zooms): median / worst |
|---|---|---|
| dense | ~1e-21 / 4e-18 | ≈ 3e-21 |
| dense_ef | ~1e-21 / 7e-19 | ≈ 4e-21 |
| fast 1e-15 | 4e-17 (interior zoom), 1–2e-16 (split / edge) / 8e-15 | **3e-14 / 9e-13** |
| fast 1e-12 | 3e-16 – 7e-15 / 1e-12 | 5e-12 |

The fast method's total noise grows linearly in n, at about n·4e-18·‖a‖₁.

**Returned length.** After a 1e-3 or 1e-6 zoom with n ≥ 64, dense returns 0.2–9% of the n+1 slots (its 1e-16 pruning). Fast always returns all n+1.

**Effect on yroots `trimMs`** (drops a trailing coefficient while Σ|tail| < 1e-3·E), with E = the transform bound only:

- `trimMs` on the fast output removes almost nothing. After a zoom of 1e-3 the fast result keeps 92–100% of its length, while the exact and dense results trim to 0.06–44% of the n+1 slots, i.e. degree ≈ 5–30.
- `trim_vs_E.csv` (zoom 1e-3) shows that trimming fast output matches exact trimming only when E ≳ 1e-8·‖a‖₁ at n = 8192, or about 1e-10 at n = 256. Roughly, E must exceed 1e3 × the noise total, about n·1e-13·‖a‖₁. Two examples:

  | n | E | fast keeps | exact keeps |
  |---|---|---|---|
  | 1024 | 1e-10·‖a‖₁ | 208 coefficients | 12 |
  | 8192 | 1e-10·‖a‖₁ | 6,976 coefficients | 29 |

- The same stall shows up along the paths in `repeated_transform.csv`, where `trimMs` was applied with E = the accumulated bound. For the first roughly 10 steps, fast output barely trims. The worst cases at n = 4096:

  | depth | exact trims to | fast keeps (worst) |
  |---|---|---|
  | 3 | 460 | ≈ 4,100 |
  | 6 | 60 | ≈ 3,000 |
  | 9 | 19 | ≈ 700 |

  Only after about 11–12 steps does E grow large relative to the noise, and fast trims like the exact result. Dense and dense_ef trim exactly like the reference throughout. In the solver this means the degree stays high over the first levels, which undercuts the fast method's speed advantage. It also means yroots' usual degree reduction does not happen until E grows.
- The noise also enters the linear-check step directly. There it acts as a floor of about 3e-14·‖a‖₁ at n = 8192 that no amount of zooming reduces.

## Bottom line

- Fast with eps = 1e-15 is about 10–40× less accurate than dense.
- It breaks the solver's per-transform bound for random-like coefficients on every split at large n: 9% of all cases, up to 22× the bound.
- The main cause is arccos(αcosθ+β) near ±1, which a cancellation-free φ formula should largely fix.
- Over deep paths, accumulated errors stay far below the accumulated bound: at most 0.03× for eps ≤ 1e-14.
- Fast leaves about u/10 of white noise in every coefficient, so dense's degree reduction after zooms disappears unless fast output is explicitly truncated. For example, drop coefficients below c·eps·‖a‖₁ and add their sum to the error, or trim to the true degree of the zoomed polynomial.
- Do not use eps = 1e-12.

## Files

- Scripts: `ddref.py`, `validate_ref.py`, `common.py`, `run_single.py`, `run_paths.py`, `run_paths_local.py`, `diag_endpoint.py`, `plots.py`, `tab.py`
- Data: `single_transform.csv`, `repeated_transform.csv`, `repeated_local_steps.csv`, `diag_endpoint.csv`, `trim_vs_E.csv`, `perindex.npz`, `summary_tables.txt`
- Plots: `single_error_vs_n.png`, `single_sup_vs_bound.png`, `error_along_index_n1024.png`, `error_along_index_n8192.png`, `accumulated_error_vs_depth.png`, `local_step_error_vs_bound.png`
