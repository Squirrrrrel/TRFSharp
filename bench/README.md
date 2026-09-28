# TRF.Sharp ↔ SciPy benchmark & validation harness

Both sides run the **same** problem set with the **same** options, so every
number in the report is an apples-to-apples comparison.

```
gen_problems.py ──▶ problems.json ──┬──▶ run_python.py   ──▶ results_python.json
                                    └──▶ csharp/BenchRunner ──▶ results_csharp.json
                                                                       │
                                              compare.py ◀─────────────┘
                                                  │
                             report.md · report.html · summary.json
```

## Running it

```bash
# 1. (re)generate the shared problem set
python gen_problems.py

# 2. SciPy side  (~40 s)
python run_python.py

# 3. TRF.Sharp side  (~2.5 min)
dotnet build -c Release csharp/BenchRunner
cd . && ./csharp/BenchRunner/bin/Release/net10.0/BenchRunner.exe results_csharp.json

# 4. compare + report
python compare.py
```

Run steps 2 and 3 **sequentially** — running them in parallel pollutes the
wall-clock timings.

## What is compared

| quantity | how |
|---|---|
| `x` | max relative deviation `|Δx_i| / max(1,|x_i|)` |
| `cost` | `|Δcost| / (1+|cost|)` (denominator-stabilised) |
| `nfev`, `njev`, `nit`, `status`, `active_mask` | exact equality — these prove the two implementations walk the same path |
| iteration trajectory | every intermediate iterate, captured through the callback of each side |

Verdicts: `IDENTICAL` (≤1e-12 + counters equal) → `MATCH` (≤1e-8) →
`MATCH*` (same answer, different path) → `EQUIV` (same cost, different `x`:
non-unique minimiser or flat drift to `max_nfev`) → `CLOSE` → `MISMATCH`.

## Files

- `gen_problems.py` — the problem catalogue (31 problems, 116 solver cases).
  Matrices travel as base64 little-endian float64 so they round-trip exactly;
  `±inf` bounds travel as the strings `"inf"`/`"-inf"` (JSON has no infinity
  literal and `System.Text.Json` rejects Python's `-Infinity`).
- `run_python.py` — `scipy.optimize.least_squares(method='trf')`.
- `csharp/BenchRunner` — same problems through `TRF.Sharp.LeastSquares.Solve`.
  Extra args: `<out.json> [name-filter] [v2]`; `v2` prints the per-iteration
  trace, which is how the `box3d` divergence below was diagnosed.
- `csharp/TrfProbe` — compiles the library **sources** (not the DLL) so it can
  call `internal` helpers; used to diff SVD outputs element by element.
- `compare.py` — verdicts, timings, `report.md`, `report.html`, `summary.json`.

## Nine-parameter focus

The target workload. Adds five nine-parameter problems (25 cases) plus a
data-count sweep of the same three-peak model at m = 32 … 1024:

| problem | n | m | what it stresses |
|---|---:|---:|---|
| `gauss3_n9` | 9 | 256 | three separated peaks — well conditioned (cond ≈ 4.6) |
| `gauss3_n9_farstart` | 9 | 256 | poor starting point, local-minimum trap |
| `gauss3_overlap_n9` | 9 | 256 | two peaks nearly coincide, cond ≈ 2.4e4 |
| `decay_peak_n9` | 9 | 220 | 2-exponential background + peak + linear baseline, cond ≈ 1.3e4 |
| `gauss3_outliers_n9` | 9 | 256 | 8% outliers, robust losses |
| `mscan_gauss3_m*` | 9 | 32…1024 | data-count sweep |

Extra tooling:

- `run_curvefit9.py` / `csharp/CurveFit9` — cross-check the public
  `CurveFit.Fit` API against `scipy.optimize.curve_fit`, including the
  covariance-derived standard errors `perr = sqrt(diag(pcov))`.
  **Note:** SciPy's `curve_fit` silently uses MINPACK `lm` when no bounds are
  given; pass `method='trf'` explicitly or you are not comparing against TRF.
- `micro_bench.py` / `csharp/Micro9` — per-kernel timings (residual, finite-
  difference Jacobian, SVD, plus what the library actually pays for a
  factorisation) on both sides.
- `csharp/SvdProbe` — accuracy and speed of thin-QR + `SVD(R)` versus
  MathNet's full SVD (three variants: managed SVD, `A.QR()`, Householder R).
- `ab_speed.py` — A/B two BenchRunner result files case by case
  (`python ab_speed.py <before.json> <after.json> [name-filter]`).
- `nine_param.py` — assembles `report_nine_param.md` from all of the above.

## Gotchas found while building this

- SciPy ≥ 1.16 no longer puts `nit` on the `trf` result — count callbacks
  instead (one per completed iteration).
- `scipy.optimize.least_squares(bounds=(lb, None))` silently turns `None` into
  `NaN`; pass `±inf` explicitly.
- SciPy's callback receives the **array** `x` unless the parameter is named
  `intermediate_result`, in which case it gets the full `OptimizeResult`.
- **MathNet's left singular vectors `U` are not accurate.** On an augmented
  Jacobian with a singular value ~1e-10 relative to the largest, `U` came out
  wrong in the 7th digit (LAPACK is exact to ~1e-16), making the Gauss-Newton
  step 4× wrong and the whole solve diverge on `box3d_n3/jac_bounded`.
  Fix: `solve_lsq_trust_region` only ever needs `suf = s .* (Uᵀ·f)`, and since
  the augmented rows carry a zero right-hand side `Aᵀ·f_aug = d·g = g_h`, so
  `suf = Vᵀ·g_h` — computed from `V`, which *is* accurate (and one mat-vec
  cheaper). See section 5 of the report for the A/B.

## Nine-parameter findings

- **Accuracy is a solved problem.** 18/25 cases match SciPy to ≤1e-8 with
  identical solver counters; with an analytic Jacobian on a well-separated
  problem the parameters agree to **5.5e-17** (machine precision). The
  remaining cases are EQUIV: the cost agrees to 1e-13…1e-18 while `x` drifts
  along a near-flat direction, which is what a non-unique minimiser looks
  like, not a porting defect.
- **Speed used to be the problem; it no longer is.** With MathNet's managed
  SVD the crossover sat at **m ≈ 64–128**: 5.4× faster at m = 32, 4× slower at
  m = 256, 100× slower at m = 1024 (112 ms vs 3 ms). `csharp/Micro9` isolated
  the cause — TRF.Sharp is *faster* at residual evaluation and at
  finite-difference Jacobians; the managed SVD is 33× slower than LAPACK
  `gesdd` at m = 256 and 209× at m = 1024 and scales worse than O(m·n²).
- **The fix is landed** (`src/TRF.Sharp/Core/ThinQr.cs`). Since `uf` is already
  obtained as `(Vᵀ·g_h)/s`, the left factor `U` is never needed, so `A = QR`
  followed by `SVD(R)` on the n×n factor reproduces `s` and `V` exactly.
  `ThinQr.FactorR` is a **Householder** QR producing **R only** — `Q` is never
  materialised — and `EconomySvd` takes that route whenever `rows ≥ 2n`.
  Speed vs the full SVD: **25× at m = 256, 154× at m = 1024, 563× at m = 4096**
  (n = 9), Gauss-Newton step unchanged to ~1e-13. MathNet's own `A.QR()` is
  *slower* than its SVD below m ≈ 64 (Gram–Schmidt, and it allocates the m×n
  `Q`); the Householder R-only version has no such crossover, so `rows ≥ 2n`
  is a correctness guard (QR needs `m ≥ n`), not a performance compromise.
- Net effect at nine parameters: geomean vs SciPy **0.28× → 1.03×** (0.27× →
  1.33× excluding `lsmr`, which never factorises). m = 256 is now at parity
  with SciPy, m = 1024 within ~2×. Across all 116 cases the geomean goes
  **0.76× → 1.43×**. Verdicts are unchanged (68 IDENTICAL, 11 EQUIV, 1
  pre-existing MISMATCH) — reproduce with
  `python ab_speed.py results_csharp_preqr.json results_csharp_qr.json`.
- What is left at large m is spread thinly over the O(m·n) MathNet
  housekeeping (`J·d`, `J·v`, norms) — tens of µs each, nothing dominant.
  Squeezing it would mean replacing MathNet's `Matrix` with raw `double[]`
  inside the solver; not worth it at a few hundred points (m = 256: 0.66 ms).
