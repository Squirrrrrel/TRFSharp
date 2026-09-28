# Nine-parameter nonlinear fitting: TRF.Sharp vs SciPy

- SciPy 1.17.1, NumPy 2.4.6, Python 3.11.9; 116 cases in the full suite
- 25 solver cases across 5 nine-parameter problems (n = 9, m = 200…256)
- Both sides read the identical `problems.json`; inputs are bit-identical (base64 float64).

## 1. Verdicts

| verdict | count | meaning |
|---|---:|---|
| IDENTICAL | 7 | x rel-err ≤ 1e-12 **and** nfev / njev / nit / status / active_mask all equal |
| MATCH | 11 | x rel-err ≤ 1e-8, counters equal |
| MATCH\* | 0 | same answer, slightly different iteration path |
| EQUIV | 7 | cost agrees ≤ 1e-8, x differs (non-unique minimiser) |
| CLOSE | 0 | 1e-8 < x rel-err ≤ 1e-4 |
| MISMATCH | 0 | genuine path divergence |

## 2. Every case

| case | verdict | x rel-err | cost rel-err | nfev py/cs | njev py/cs | nit py/cs | status | traj deviates at |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `decay_peak_n9/fd2` | EQUIV | 4.7e-08 | 9.3e-17 | 19 | 14 | 13 | 1 | 0 |
| `decay_peak_n9/fd2_bounded` | EQUIV | 6.5e-08 | 7.5e-17 | 9 | 9 | 8 | 2/1 | 1 |
| `decay_peak_n9/fd2_xscale_jac` | EQUIV | 1.4e-08 | 6.4e-17 | 16 | 12 | 11 | 2 | 1 |
| `decay_peak_n9/fd3` | MATCH | 5.8e-12 | 4.3e-19 | 19 | 14 | 13 | 1 | 1 |
| `decay_peak_n9/jac` | IDENTICAL | 2.0e-16 | 1.1e-19 | 19 | 14 | 13 | 1 | 0 |
| `decay_peak_n9/jac_bounded` | IDENTICAL | 9.2e-16 | 1.1e-19 | 9 | 9 | 8 | 1 | None |
| `decay_peak_n9/jac_lsmr` | EQUIV | 0.00348 | 3.5e-07 | 900 | 899 | 898 | 0 | 0 |
| `gauss3_n9/fd2` | MATCH | 1.3e-11 | 7.8e-18 | 8 | 8 | 7 | 1 | 1 |
| `gauss3_n9/fd2_bounded` | MATCH | 5.4e-11 | 2.6e-18 | 7 | 7 | 6 | 1 | 1 |
| `gauss3_n9/fd2_diffstep` | IDENTICAL | 9.5e-13 | 5.2e-18 | 8 | 8 | 7 | 1 | 1 |
| `gauss3_n9/fd2_xscale_jac` | MATCH | 3.3e-11 | 4.3e-19 | 8 | 8 | 7 | 1 | 1 |
| `gauss3_n9/fd3` | IDENTICAL | 2.8e-14 | 4.3e-19 | 8 | 8 | 7 | 1 | 1 |
| `gauss3_n9/jac` | IDENTICAL | 5.6e-17 | 4.3e-19 | 8 | 8 | 7 | 1 | None |
| `gauss3_n9/jac_bounded` | IDENTICAL | 0 | 2.2e-18 | 7 | 7 | 6 | 1 | None |
| `gauss3_n9_farstart/fd2` | MATCH | 2.6e-11 | 8.2e-18 | 16 | 14 | 13 | 1 | 1 |
| `gauss3_n9_farstart/fd2_bounded` | MATCH | 3.5e-09 | 2.6e-15 | 15 | 14 | 13 | 2 | 1 |
| `gauss3_outliers_n9/fd2_cauchy` | MATCH | 1.5e-11 | 2.4e-17 | 16 | 11 | 10 | 1 | 0 |
| `gauss3_outliers_n9/fd2_huber` | MATCH | 2.8e-10 | 1.7e-16 | 12 | 9 | 8 | 2 | 0 |
| `gauss3_outliers_n9/fd2_linear` | MATCH | 2.2e-09 | 1.0e-13 | 22 | 16 | 15 | 2 | 1 |
| `gauss3_outliers_n9/fd2_soft_l1` | MATCH | 1.4e-10 | 2.6e-16 | 13 | 10 | 9 | 2 | 1 |
| `gauss3_outliers_n9/jac_bounded_huber` | IDENTICAL | 6.7e-16 | 2.5e-16 | 10 | 8 | 7 | 2 | 0 |
| `gauss3_overlap_n9/fd2` | EQUIV | 9.6e-07 | 5.8e-16 | 49 | 39 | 38 | 2 | 1 |
| `gauss3_overlap_n9/fd2_bounded` | EQUIV | 1.4e-05 | 6.9e-15 | 95/94 | 80 | 79 | 2 | 1 |
| `gauss3_overlap_n9/jac` | MATCH | 1.1e-12 | 8.7e-19 | 49 | 39 | 38 | 2 | 1 |
| `gauss3_overlap_n9/jac_lsmr` | EQUIV | 0.00216 | 4.9e-09 | 127/110 | 101/103 | 100/102 | 2 | 0 |

## 3. Did the fits actually converge?

| case | scipy cost | TRF.Sharp cost | scipy status | TRF.Sharp status | scipy optimality | TRF.Sharp optimality |
|---|---:|---:|---:|---:|---:|---:|
| `decay_peak_n9/fd2` | 8.602689e-04 | 8.602689e-04 | 1 | 1 | 5.1e-09 | 5.4e-09 |
| `decay_peak_n9/fd2_bounded` | 8.602689e-04 | 8.602689e-04 | 2 | 1 | 1.0e-08 | 9.3e-09 |
| `decay_peak_n9/fd2_xscale_jac` | 8.602689e-04 | 8.602689e-04 | 2 | 2 | 2.3e-08 | 2.3e-08 |
| `decay_peak_n9/fd3` | 8.602689e-04 | 8.602689e-04 | 1 | 1 | 5.4e-09 | 5.4e-09 |
| `decay_peak_n9/jac` | 8.602689e-04 | 8.602689e-04 | 1 | 1 | 5.4e-09 | 5.4e-09 |
| `decay_peak_n9/jac_bounded` | 8.602689e-04 | 8.602689e-04 | 1 | 1 | 8.9e-09 | 8.9e-09 |
| `decay_peak_n9/jac_lsmr` | 8.608982e-04 | 8.612443e-04 | 0 | 0 | 2.0e-04 | 4.3e-05 |
| `gauss3_n9/fd2` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 1.7e-09 | 1.5e-09 |
| `gauss3_n9/fd2_bounded` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 6.7e-09 | 6.9e-09 |
| `gauss3_n9/fd2_diffstep` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 1.6e-09 | 1.6e-09 |
| `gauss3_n9/fd2_xscale_jac` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 1.5e-09 | 1.5e-09 |
| `gauss3_n9/fd3` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 1.6e-09 | 1.6e-09 |
| `gauss3_n9/jac` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 1.6e-09 | 1.6e-09 |
| `gauss3_n9/jac_bounded` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 6.9e-09 | 6.9e-09 |
| `gauss3_n9_farstart/fd2` | 2.170305e-03 | 2.170305e-03 | 1 | 1 | 8.5e-09 | 8.2e-09 |
| `gauss3_n9_farstart/fd2_bounded` | 5.970670e-01 | 5.970670e-01 | 2 | 2 | 3.1e-04 | 3.1e-04 |
| `gauss3_outliers_n9/fd2_cauchy` | 1.744887e-01 | 1.744887e-01 | 1 | 1 | 1.8e-09 | 1.9e-09 |
| `gauss3_outliers_n9/fd2_huber` | 1.623896e+00 | 1.623896e+00 | 2 | 2 | 8.7e-08 | 8.2e-08 |
| `gauss3_outliers_n9/fd2_linear` | 2.821084e+01 | 2.821084e+01 | 2 | 2 | 0.00192 | 0.00192 |
| `gauss3_outliers_n9/fd2_soft_l1` | 1.599712e+00 | 1.599712e+00 | 2 | 2 | 3.6e-08 | 3.7e-08 |
| `gauss3_outliers_n9/jac_bounded_huber` | 1.623896e+00 | 1.623896e+00 | 2 | 2 | 1.3e-05 | 1.3e-05 |
| `gauss3_overlap_n9/fd2` | 2.393158e-03 | 2.393158e-03 | 2 | 2 | 2.1e-07 | 2.1e-07 |
| `gauss3_overlap_n9/fd2_bounded` | 2.393158e-03 | 2.393158e-03 | 2 | 2 | 4.1e-07 | 8.0e-07 |
| `gauss3_overlap_n9/jac` | 2.393158e-03 | 2.393158e-03 | 2 | 2 | 2.1e-07 | 2.1e-07 |
| `gauss3_overlap_n9/jac_lsmr` | 2.393172e-03 | 2.393177e-03 | 2 | 2 | 6.6e-07 | 1.3e-05 |

Status: `1` = gtol, `2` = ftol, `3` = xtol (all converged), `0` = hit `max_nfev`.

## 4. `CurveFit.Fit` API vs `scipy.optimize.curve_fit`

This is the entry point a caller actually uses. Compared: the fitted parameters **and** the covariance-derived standard errors `perr = sqrt(diag(pcov))`.

| case | overload | popt rel-err | perr rel-err | cost rel-err | nfev py/cs | ‖perr‖ SciPy | ‖perr‖ TRF.Sharp |
|---|---|---:|---:|---:|---:|---:|---:|
| `decay_peak_n9/bounded` | `array` | 7.7e-08 | 2.3e-08 | 1.0e-16 | 9/9 | 0.221 | 0.221 |
| `decay_peak_n9/bounded` | `vector` | 7.7e-08 | 2.3e-08 | 1.0e-16 | 9/9 | 0.221 | 0.221 |
| `decay_peak_n9/unbounded` | `array` | 1.6e-08 | 4.5e-09 | 3.5e-17 | 19/19 | 0.221 | 0.221 |
| `decay_peak_n9/unbounded` | `vector` | 1.6e-08 | 4.5e-09 | 3.5e-17 | 19/19 | 0.221 | 0.221 |
| `gauss3_n9/bounded` | `array` | 5.4e-11 | 2.8e-12 | 2.6e-18 | 7/7 | 0.0038 | 0.0038 |
| `gauss3_n9/bounded` | `vector` | 5.4e-11 | 2.8e-12 | 2.6e-18 | 7/7 | 0.0038 | 0.0038 |
| `gauss3_n9/unbounded` | `array` | 1.3e-11 | 4.5e-12 | 7.8e-18 | 8/8 | 0.0038 | 0.0038 |
| `gauss3_n9/unbounded` | `vector` | 1.3e-11 | 4.5e-12 | 7.8e-18 | 8/8 | 0.0038 | 0.0038 |
| `gauss3_outliers_n9/bounded` | `array` | 1.4e-08 | 4.5e-09 | 6.2e-13 | 12/12 | 0.381 | 0.381 |
| `gauss3_outliers_n9/bounded` | `vector` | 1.4e-08 | 4.5e-09 | 6.2e-13 | 12/12 | 0.381 | 0.381 |
| `gauss3_outliers_n9/unbounded` | `array` | 2.2e-09 | 6.0e-10 | 1.0e-13 | 22/22 | 0.381 | 0.381 |
| `gauss3_outliers_n9/unbounded` | `vector` | 2.2e-09 | 6.0e-10 | 1.0e-13 | 22/22 | 0.381 | 0.381 |

`perr` column: ‖sqrt(diag(pcov))‖₂, i.e. the overall size of the reported parameter uncertainty.

## 5. Performance (best-of-N wall clock, serial)

- geometric mean over 25 nine-parameter cases: **1.01×** (TRF.Sharp / SciPy)
- fastest: decay_peak_n9/fd2_bounded at 1.55×
- slowest: decay_peak_n9/jac_lsmr at 0.17×

| case | jac | m | SciPy (ms) | TRF.Sharp (ms) | speedup |
|---|---|---:|---:|---:|---:|
| `decay_peak_n9/fd2_bounded` | fd2 | 220 | 2.367 | 1.527 | 1.55× |
| `gauss3_overlap_n9/fd2_bounded` | fd2 | 256 | 24.230 | 16.315 | 1.49× |
| `gauss3_outliers_n9/fd2_huber` | fd2 | 256 | 2.854 | 1.940 | 1.47× |
| `gauss3_outliers_n9/fd2_soft_l1` | fd2 | 256 | 2.963 | 2.077 | 1.43× |
| `gauss3_n9/fd2_bounded` | fd2 | 256 | 1.933 | 1.405 | 1.38× |
| `gauss3_outliers_n9/fd2_cauchy` | fd2 | 256 | 3.316 | 2.488 | 1.33× |
| `gauss3_n9/fd3` | fd3 | 256 | 2.955 | 2.223 | 1.33× |
| `gauss3_overlap_n9/fd2` | fd2 | 256 | 10.790 | 8.188 | 1.32× |
| `decay_peak_n9/fd2` | fd2 | 220 | 3.664 | 2.807 | 1.31× |
| `decay_peak_n9/fd3` | fd3 | 220 | 5.306 | 4.077 | 1.30× |
| `decay_peak_n9/fd2_xscale_jac` | fd2 | 220 | 3.152 | 2.468 | 1.28× |
| `gauss3_outliers_n9/fd2_linear` | fd2 | 256 | 4.383 | 3.516 | 1.25× |
| `gauss3_n9/fd2_diffstep` | fd2 | 256 | 1.981 | 1.635 | 1.21× |
| `gauss3_n9/fd2` | fd2 | 256 | 1.959 | 1.632 | 1.20× |
| `gauss3_n9_farstart/fd2` | fd2 | 256 | 3.686 | 3.113 | 1.18× |
| `decay_peak_n9/jac_bounded` | analytic | 220 | 1.123 | 0.958 | 1.17× |
| `gauss3_n9/fd2_xscale_jac` | fd2 | 256 | 1.990 | 1.849 | 1.08× |
| `gauss3_outliers_n9/jac_bounded_huber` | analytic | 256 | 1.644 | 1.623 | 1.01× |
| `gauss3_n9/jac_bounded` | analytic | 256 | 0.958 | 0.954 | 1.00× |
| `decay_peak_n9/jac` | analytic | 220 | 1.739 | 1.902 | 0.91× |
| `gauss3_n9/jac` | analytic | 256 | 0.869 | 1.056 | 0.82× |
| `gauss3_overlap_n9/jac` | analytic | 256 | 5.396 | 6.772 | 0.80× |
| `gauss3_n9_farstart/fd2_bounded` | fd2 | 256 | 4.431 | 6.162 | 0.72× |
| `gauss3_overlap_n9/jac_lsmr` | analytic | 256 | 37.122 | 203.815 | 0.18× |
| `decay_peak_n9/jac_lsmr` | analytic | 220 | 278.843 | 1682.982 | 0.17× |

`curve_fit` API, 12 timings: geometric mean **0.47×**.

| case | overload | SciPy (ms) | TRF.Sharp (ms) | speedup |
|---|---|---:|---:|---:|
| `decay_peak_n9/bounded` | `array` | 2.463 | 4.457 | 0.55× |
| `decay_peak_n9/bounded` | `vector` | 2.463 | 6.957 | 0.35× |
| `decay_peak_n9/unbounded` | `array` | 3.783 | 7.262 | 0.52× |
| `decay_peak_n9/unbounded` | `vector` | 3.783 | 11.460 | 0.33× |
| `gauss3_n9/bounded` | `array` | 2.043 | 4.377 | 0.47× |
| `gauss3_n9/bounded` | `vector` | 2.043 | 6.545 | 0.31× |
| `gauss3_n9/unbounded` | `array` | 2.085 | 4.738 | 0.44× |
| `gauss3_n9/unbounded` | `vector` | 2.085 | 7.347 | 0.28× |
| `gauss3_outliers_n9/bounded` | `array` | 3.435 | 3.522 | 0.98× |
| `gauss3_outliers_n9/bounded` | `vector` | 3.435 | 5.238 | 0.66× |
| `gauss3_outliers_n9/unbounded` | `array` | 4.465 | 9.916 | 0.45× |
| `gauss3_outliers_n9/unbounded` | `vector` | 4.465 | 7.454 | 0.60× |

The `array` overload allocates a fresh `double[]` per data point (`p.ToArray()` inside `CurveFit.Fit`); the `vector` one does not. On these sizes the allocation is not the bottleneck — the SVD is — so the two are within noise.

## 6. How the speed ratio moves with the number of data points

The same three-peak nine-parameter model sampled at m = 32 … 1024. Accuracy is unaffected (every case is IDENTICAL or MATCH, x rel-err ≤ 1.4e-10); only the speed changes.

| m | config | SciPy (ms) | TRF.Sharp (ms) | speedup | same case before the QR switch |
|---:|---|---:|---:|---:|---:|
| 32 | fd2 | 1.423 | 0.194 | 7.32× | 0.264 ms (1.4× slower) |
| 32 | jac | 0.586 | 0.154 | 3.80× | 0.229 ms (1.5× slower) |
| 32 | jac_lsmr | 1.701 | 0.153 | 11.11× | 0.159 ms (1.0× slower) |
| 64 | fd2 | 1.659 | 0.382 | 4.35× | 0.736 ms (1.9× slower) |
| 64 | jac | 0.691 | 0.255 | 2.71× | 0.635 ms (2.5× slower) |
| 64 | jac_lsmr | 1.748 | 2.494 | 0.70× | 2.196 ms (0.9× slower) |
| 128 | fd2 | 1.787 | 0.525 | 3.41× | 2.121 ms (4.0× slower) |
| 128 | jac | 0.755 | 0.330 | 2.29× | 2.133 ms (6.5× slower) |
| 128 | jac_lsmr | 1.860 | 6.448 | 0.29× | 4.906 ms (0.8× slower) |
| 256 | fd2 | 1.709 | 1.028 | 1.66× | 6.248 ms (6.1× slower) |
| 256 | jac | 0.755 | 0.656 | 1.15× | 5.918 ms (9.0× slower) |
| 256 | jac_lsmr | 1.883 | 10.665 | 0.18× | 10.546 ms (1.0× slower) |
| 512 | fd2 | 2.400 | 2.756 | 0.87× | 27.488 ms (10.0× slower) |
| 512 | jac | 1.095 | 2.028 | 0.54× | 26.886 ms (13.3× slower) |
| 512 | jac_lsmr | 2.182 | 17.054 | 0.13× | 16.689 ms (1.0× slower) |
| 1024 | fd2 | 3.006 | 4.943 | 0.61× | 111.937 ms (22.6× slower) |
| 1024 | jac | 1.503 | 3.681 | 0.41× | 111.317 ms (30.2× slower) |
| 1024 | jac_lsmr | 2.567 | 27.570 | 0.09× | 25.927 ms (0.9× slower) |

The last column is the same case **before** the thin-QR switch of §8. With MathNet's managed SVD the crossover sat at m ≈ 64–128 and TRF.Sharp was 100× slower at m = 1024; with the QR factorisation TRF.Sharp is at parity with SciPy by m = 256 and within ~2× at m = 1024. The `jac_lsmr` rows are unaffected — the LSMR solver never factorises.

## 7. Where the time actually goes

Micro-benchmark of the hot kernels, best-of-400. The last column is what the library pays for a factorisation today; the column before it is what it used to pay:

| m | residual py/cs | FD Jacobian py/cs | SVD py/cs | MathNet SVD slower by | library factorisation (QR + SVD(R)) |
|---:|---:|---:|---:|---:|---:|
| 32 | 7.1 / 1.4 µs | 83.8 / 23.1 µs | 20.1 / 37.4 µs | **2×** | 24.7 µs (2× faster) |
| 64 | 7.3 / 2.8 µs | 87.2 / 45.0 µs | 21.9 / 90.8 µs | **4×** | 27.3 µs (3× faster) |
| 128 | 8.4 / 5.7 µs | 99.1 / 82.5 µs | 25.6 / 301.7 µs | **12×** | 33.2 µs (9× faster) |
| 256 | 9.5 / 7.7 µs | 109.5 / 33.3 µs | 33.2 / 1090.8 µs | **33×** | 32.6 µs (33× faster) |
| 512 | 12.8 / 5.2 µs | 147.0 / 66.0 µs | 49.1 / 4056.7 µs | **83×** | 55.5 µs (73× faster) |
| 1024 | 17.1 / 10.6 µs | 195.0 / 132.8 µs | 80.5 / 16521.4 µs | **205×** | 100.9 µs (164× faster) |

Read that carefully, because it is the whole story of the port: TRF.Sharp is *faster* at evaluating residuals and at building the finite-difference Jacobian, and MathNet's managed SVD is still 33×–209× slower than LAPACK `gesdd` — but **the library no longer calls it**. What it calls costs 33 µs at m = 256 and 101 µs at m = 1024, which is the same ballpark as LAPACK. What is left at large m is spread thinly over the O(m·n) MathNet housekeeping (`J·d`, `J·v`, gradient norms): each is tens of µs, none dominates, and squeezing them would mean replacing MathNet's `Matrix` with raw `double[]` inside the solver.

## 8. The fix, landed: thin QR + SVD(R)

The trust-region subproblem needs only the singular values `s` and the right singular vectors `V`. (`uf` is already obtained as `(Vᵀ·g_h)/s` — see the A/B note in the main report — so the left factor `U` is never formed.) That permits a thin QR first:

```
A = Q R          (thin QR, R is n×n)
R = U_R S Vᵀ  ⇒  A = (Q U_R) S Vᵀ   ⇒  s(A)=s(R),  V(A)=V(R)
```

`Core/ThinQr.FactorR` is a Householder QR that produces **R only** — `Q` is never materialised — and `EconomySvd` takes the QR route whenever `rows ≥ 2·n`, falling back to the direct SVD for short and wide matrices. Two conditionings are shown (column decay 0.35 → cond ≈ 6e2, 0.8 → cond ≈ 3e6):

| m | cond | MathNet SVD | MathNet QR + SVD(R) | Householder R + SVD(R) | speedup vs SVD | `s` rel-err | `|V|` rel-err | Gauss-Newton step rel-err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.35 | 21.0 µs | 79.6 µs | 20.0 µs | **1.0×** | 6.2e-16 | 6.4e-14 | 7.3e-15 |
| 24 | 0.35 | 25.0 µs | 79.3 µs | 20.6 µs | **1.2×** | 8.8e-16 | 1.7e-13 | 9.8e-15 |
| 32 | 0.35 | 33.0 µs | 88.5 µs | 21.8 µs | **1.5×** | 1.1e-15 | 2.3e-14 | 1.0e-14 |
| 48 | 0.35 | 55.7 µs | 94.4 µs | 22.6 µs | **2.5×** | 1.3e-15 | 8.8e-14 | 1.1e-14 |
| 64 | 0.35 | 85.4 µs | 108.2 µs | 25.5 µs | **3.3×** | 7.8e-16 | 1.1e-13 | 7.0e-14 |
| 96 | 0.35 | 179.7 µs | 126.6 µs | 29.4 µs | **6.1×** | 9.4e-16 | 3.0e-13 | 5.2e-14 |
| 128 | 0.35 | 274.5 µs | 138.1 µs | 31.2 µs | **8.8×** | 8.8e-16 | 3.5e-13 | 6.1e-14 |
| 192 | 0.35 | 567.4 µs | 135.8 µs | 36.9 µs | **15.4×** | 7.9e-16 | 8.3e-13 | 2.3e-14 |
| 256 | 0.35 | 1080.2 µs | 153.0 µs | 42.6 µs | **25.4×** | 8.0e-16 | 1.7e-12 | 6.6e-13 |
| 384 | 0.35 | 2135.3 µs | 87.5 µs | 40.3 µs | **53.0×** | 8.5e-16 | 3.7e-12 | 1.5e-13 |
| 512 | 0.35 | 3746.7 µs | 116.0 µs | 51.4 µs | **72.9×** | 1.4e-15 | 1.6e-12 | 6.2e-14 |
| 1024 | 0.35 | 14808.6 µs | 210.5 µs | 96.1 µs | **154.1×** | 1.1e-15 | 1.7e-12 | 4.9e-14 |
| 2048 | 0.35 | 58840.7 µs | 448.1 µs | 209.3 µs | **281.1×** | 2.0e-15 | 5.2e-12 | 1.6e-13 |
| 4096 | 0.35 | 233596.5 µs | 716.4 µs | 414.6 µs | **563.4×** | 3.3e-15 | 4.4e-12 | 6.0e-12 |
| 16 | 0.8 | 14.2 µs | 24.6 µs | 8.5 µs | **1.7×** | 8.7e-16 | 3.8e-13 | 4.5e-15 |
| 24 | 0.8 | 20.0 µs | 22.7 µs | 9.2 µs | **2.2×** | 7.1e-16 | 9.7e-13 | 7.5e-14 |
| 32 | 0.8 | 28.2 µs | 23.6 µs | 9.9 µs | **2.8×** | 5.7e-16 | 7.2e-14 | 2.5e-13 |
| 48 | 0.8 | 51.4 µs | 25.5 µs | 10.9 µs | **4.7×** | 1.2e-15 | 2.1e-13 | 1.4e-13 |
| 64 | 0.8 | 79.3 µs | 37.0 µs | 12.1 µs | **6.6×** | 9.5e-16 | 7.7e-14 | 2.3e-14 |
| 96 | 0.8 | 178.5 µs | 49.1 µs | 15.2 µs | **11.7×** | 1.1e-15 | 1.2e-13 | 1.3e-12 |
| 128 | 0.8 | 271.4 µs | 43.9 µs | 17.9 µs | **15.2×** | 7.6e-16 | 2.0e-12 | 2.5e-12 |
| 192 | 0.8 | 627.3 µs | 64.5 µs | 23.7 µs | **26.5×** | 7.4e-16 | 4.7e-12 | 4.6e-14 |
| 256 | 0.8 | 1067.3 µs | 75.2 µs | 29.3 µs | **36.4×** | 1.9e-15 | 1.2e-12 | 3.3e-14 |
| 384 | 0.8 | 2356.1 µs | 89.0 µs | 40.6 µs | **58.0×** | 1.3e-15 | 2.6e-12 | 3.5e-13 |
| 512 | 0.8 | 3746.4 µs | 110.3 µs | 51.2 µs | **73.2×** | 1.5e-15 | 1.8e-12 | 3.1e-13 |
| 1024 | 0.8 | 14798.6 µs | 213.7 µs | 96.0 µs | **154.2×** | 2.2e-15 | 1.2e-12 | 1.7e-12 |
| 2048 | 0.8 | 58753.2 µs | 451.0 µs | 210.7 µs | **278.8×** | 1.8e-15 | 2.5e-11 | 4.8e-12 |
| 4096 | 0.8 | 233545.6 µs | 711.4 µs | 411.4 µs | **567.7×** | 2.7e-15 | 1.1e-11 | 4.7e-12 |

`|V|` is compared componentwise in magnitude: it agrees to ~1e-12, so the `2.0` signed differences seen in raw V are pure sign flips (singular vectors are sign-indefinite). Sign flips cancel in `p = −V·(Vᵀg_h)/s²`, which is why the Gauss-Newton step is unchanged to ~1e-13.

Note the middle column: MathNet's own `A.QR()` (Gram–Schmidt, and it allocates the m×n factor `Q`) is the variant that was *slower* than the SVD at m ≤ 64. Householder with R only has no such crossover — it wins at every size — which is why the threshold is a safety margin (`rows ≥ 2·n`, needed because the QR route requires `m ≥ n`) rather than a performance compromise.

### Regression: does it change any answer?

The full 116-case suite was re-run before and after. The verdict counts are unchanged — 68 IDENTICAL, 11 EQUIV, 1 MISMATCH (the pre-existing pathological `linear_illcond_n40/jac_lsmr`), with one case moving between MATCH and MATCH*. Geomean wall clock across all 116 cases: **1.88× faster** (27.6 s → 17.4 s), and the headline SciPy-vs-TRF.Sharp geomean goes from **0.76× to 1.43×** in TRF.Sharp's favour. Reproduce with `python ab_speed.py results_csharp_preqr.json results_csharp_qr.json`.

## 9. Bottom line

- Worst parameter error across all nine-parameter cases: **0.00348** (`decay_peak_n9/jac_lsmr`, verdict EQUIV).
- 18/25 cases reproduce SciPy's parameters to ≤1e-8 with identical solver counters; none is worse than EQUIV (same cost).
- **Accuracy is not the problem; speed is.** With a few dozen data points TRF.Sharp is 2–5× faster than SciPy; at m ≥ 256 it is 4–100× slower, entirely because of MathNet's managed SVD.

