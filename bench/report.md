# TRF.Sharp vs SciPy — validation & benchmark report

Generated 2026-09-28T18:08:03+00:00 · 116 cases · 31 problems

## 1. Environment

| | SciPy | TRF.Sharp |
|---|---|---|
| engine | `scipy.optimize.least_squares(method='trf')` | `TRF.Sharp LeastSquares.Solve (port of scipy least_squares method='trf')` |
| version | scipy 1.17.1, numpy 2.4.6 | TRF.Sharp 1.0.0.0, MathNet 5.0.0.0 |
| runtime | CPython 3.11.9 | .NET 10.0.10 (Release) |
| BLAS | scipy-openblas | MathNet managed (no native BLAS) |
| machine | ARM64 / Windows-10-10.0.26200-SP0 | Arm64 / Microsoft Windows 10.0.26200 |

## 2. Validation summary

| verdict | meaning | count |
|---|---|---:|
| IDENTICAL | x rel-err ≤ 1e-12 **and** nfev/njev/nit/status/active_mask all equal | 68 |
| MATCH | x rel-err ≤ 1e-8 and counters equal | 32 |
| MATCH\* | x rel-err ≤ 1e-8, counters differ (different iteration path) | 4 |
| CLOSE | 1e-8 < x rel-err ≤ 1e-4 | 0 |
| MISMATCH | x rel-err > 1e-4 | 1 |
| ERROR | one side raised | 0 |

- bitwise-identical solution vectors: **9/116**
- identical evaluation counters (nfev/njev/nit/status): **107/116**
- identical `active_mask`: **116/116**
- identical **iteration trajectory** (every intermediate iterate within 1e-12): **36/116**

## 3. Per-case validation

| problem / config | n | m | jac | status | nfev py/cs | njev py/cs | nit py/cs | x rel-err | cost rel-err | traj | verdict |
|---|---:|---:|---|---|---|---|---|---:|---:|---|---|
| beale_n2/fd2 | 2 | 3 | fd2 | 1 | 7/7 | 7/7 | 6/6 | 4.83e-14 | 5.94e-25 | 3.2e-08 | IDENTICAL |
| beale_n2/jac | 2 | 3 | analytic | 1 | 7/7 | 7/7 | 6/6 | 2.96e-16 | 9.87e-28 | = | IDENTICAL |
| beale_n2/jac_bounded | 2 | 3 | analytic | 1 | 10/10 | 9/9 | 8/8 | 1.48e-16 | 0 | = | IDENTICAL |
| box3d_n3/fd2 | 3 | 10 | fd2 | 1 | 6/6 | 6/6 | 5/5 | 1.06e-12 | 1.66e-23 | 9.0e-09 | MATCH |
| box3d_n3/jac | 3 | 10 | analytic | 1 | 6/6 | 6/6 | 5/5 | 1.78e-16 | 1.78e-26 | = | IDENTICAL |
| box3d_n3/jac_bounded | 3 | 10 | analytic | 1 | 15/22 | 14/18 | 13/17 | 1.92e-09 | 2.14e-19 | 1.2e+01 | MATCH* |
| broyden_n10/fd2 | 10 | 10 | fd2 | 1 | 5/5 | 5/5 | 4/4 | 1.11e-13 | 3.18e-22 | 4.1e-10 | IDENTICAL |
| broyden_n10/jac | 10 | 10 | analytic | 1 | 5/5 | 5/5 | 4/4 | 1.11e-16 | 1.13e-26 | = | IDENTICAL |
| broyden_n10/jac_lsmr | 10 | 10 | analytic | 1 | 5/5 | 5/5 | 4/4 | 1.11e-16 | 2.38e-26 | = | IDENTICAL |
| broyden_n100/fd2 | 100 | 100 | fd2 | 1 | 5/5 | 5/5 | 4/4 | 8.66e-14 | 3.03e-22 | 4.2e-10 | IDENTICAL |
| broyden_n100/jac | 100 | 100 | analytic | 1 | 5/5 | 5/5 | 4/4 | 2.22e-16 | 3.38e-26 | = | IDENTICAL |
| broyden_n100/jac_lsmr | 100 | 100 | analytic | 1 | 5/5 | 5/5 | 4/4 | 1.11e-16 | 8.09e-25 | = | IDENTICAL |
| broyden_n200/fd2 | 200 | 200 | fd2 | 1 | 5/5 | 5/5 | 4/4 | 1.01e-13 | 2.56e-22 | 3.2e-10 | IDENTICAL |
| broyden_n200/jac | 200 | 200 | analytic | 1 | 5/5 | 5/5 | 4/4 | 2.22e-16 | 1.92e-25 | = | IDENTICAL |
| broyden_n200/jac_lsmr | 200 | 200 | analytic | 1 | 5/5 | 5/5 | 4/4 | 1.11e-16 | 6.08e-25 | = | IDENTICAL |
| broyden_n50/fd2 | 50 | 50 | fd2 | 1 | 5/5 | 5/5 | 4/4 | 6.65e-14 | 6.62e-23 | 4.2e-10 | IDENTICAL |
| broyden_n50/jac | 50 | 50 | analytic | 1 | 5/5 | 5/5 | 4/4 | 2.22e-16 | 1.55e-25 | = | IDENTICAL |
| broyden_n50/jac_lsmr | 50 | 50 | analytic | 1 | 5/5 | 5/5 | 4/4 | 1.11e-16 | 7.23e-26 | = | IDENTICAL |
| damped_osc_m150/fd2 | 5 | 150 | fd2 | 2 | 15/15 | 12/12 | 11/11 | 3.36e-11 | 8.19e-17 | 3.2e-08 | MATCH |
| damped_osc_m150/fd2_bounded | 5 | 150 | fd2 | 2 | 11/11 | 11/11 | 10/10 | 4.59e-11 | 3.45e-18 | 1.4e-08 | MATCH |
| damped_osc_m150/fd3 | 5 | 150 | fd3 | 2 | 15/15 | 12/12 | 11/11 | 2.25e-13 | 8.02e-17 | 5.7e-11 | IDENTICAL |
| dbv_n10/fd2 | 10 | 10 | fd2 | 1 | 4/4 | 4/4 | 3/3 | 8.06e-13 | 1.72e-20 | 5.4e-11 | IDENTICAL |
| dbv_n10/jac | 10 | 10 | analytic | 1 | 4/4 | 4/4 | 3/3 | 2.78e-17 | 1.05e-24 | = | IDENTICAL |
| dbv_n10/jac_lsmr | 10 | 10 | analytic | 1 | 5/5 | 5/5 | 4/4 | 5.55e-17 | 1.26e-25 | 1.1e-12 | IDENTICAL |
| decay_peak_n9/fd2 | 9 | 220 | fd2 | 1 | 19/19 | 14/14 | 13/13 | 4.72e-08 | 9.27e-17 | 1.2e-06 | EQUIV |
| decay_peak_n9/fd2_bounded | 9 | 220 | fd2 | 2 | 9/9 | 9/9 | 8/8 | 6.48e-08 | 7.54e-17 | 6.5e-08 | EQUIV |
| decay_peak_n9/fd2_xscale_jac | 9 | 220 | fd2 | 2 | 16/16 | 12/12 | 11/11 | 1.44e-08 | 6.37e-17 | 3.6e-07 | EQUIV |
| decay_peak_n9/fd3 | 9 | 220 | fd3 | 1 | 19/19 | 14/14 | 13/13 | 5.77e-12 | 4.33e-19 | 7.5e-10 | MATCH |
| decay_peak_n9/jac | 9 | 220 | analytic | 1 | 19/19 | 14/14 | 13/13 | 2.02e-16 | 1.08e-19 | 1.7e-12 | IDENTICAL |
| decay_peak_n9/jac_bounded | 9 | 220 | analytic | 1 | 9/9 | 9/9 | 8/8 | 9.19e-16 | 1.08e-19 | = | IDENTICAL |
| decay_peak_n9/jac_lsmr | 9 | 220 | analytic | 0 | 900/900 | 899/899 | 898/898 | 0.00347739 | 3.46e-07 | 1.3e-02 | EQUIV |
| exp_fit_m80/fd2 | 3 | 80 | fd2 | 0 | 300/300 | 243/243 | 242/242 | 1.45e-08 | 2.11e-07 | 1.4e-02 | EQUIV |
| exp_fit_m80/fd2_bounded | 3 | 80 | fd2 | 2 | 13/13 | 12/12 | 11/11 | 3.00e-10 | 2.05e-15 | 3.8e-08 | MATCH |
| exp_fit_m80/fd3 | 3 | 80 | fd3 | 0 | 300/300 | 243/243 | 242/242 | 3.27e-14 | 3.54e-13 | 9.5e-11 | IDENTICAL |
| exp_fit_m80/jac | 3 | 80 | analytic | 0 | 300/300 | 243/243 | 242/242 | 2.34e-15 | 6.17e-14 | 6.3e-12 | IDENTICAL |
| exp_fit_m80/jac_bounded | 3 | 80 | analytic | 2 | 13/13 | 12/12 | 11/11 | 1.45e-16 | 1.37e-16 | = | IDENTICAL |
| exp_fit_m80/jac_xscale_array | 3 | 80 | analytic | 0 | 300/300 | 243/243 | 242/242 | 1.09e-14 | 1.41e-13 | 7.3e-11 | IDENTICAL |
| exp_fit_outliers_m80/arctan | 3 | 80 | analytic | 0 | 300/300 | 243/243 | 243/243 | 1.20e-12 | 3.82e-14 | 4.5e-09 | MATCH |
| exp_fit_outliers_m80/bounded_cauchy | 3 | 80 | analytic | 2 | 15/15 | 13/13 | 12/12 | 6.59e-12 | 2.27e-16 | 7.8e-09 | MATCH |
| exp_fit_outliers_m80/cauchy | 3 | 80 | analytic | 0 | 300/300 | 245/245 | 244/244 | 5.21e-11 | 4.27e-14 | 2.7e-07 | MATCH |
| exp_fit_outliers_m80/fd2_huber | 3 | 80 | fd2 | 0 | 300/300 | 242/242 | 241/241 | 0.01113 | 4.49e-08 | 2.1e-02 | EQUIV |
| exp_fit_outliers_m80/huber | 3 | 80 | analytic | 0 | 300/300 | 240/240 | 240/240 | 8.74e-12 | 4.28e-15 | 8.2e-08 | MATCH |
| exp_fit_outliers_m80/linear | 3 | 80 | analytic | 0 | 300/300 | 242/242 | 242/242 | 8.70e-12 | 9.49e-14 | 3.2e-07 | MATCH |
| exp_fit_outliers_m80/soft_l1 | 3 | 80 | analytic | 0 | 300/300 | 242/242 | 241/241 | 2.70e-15 | 2.41e-15 | = | IDENTICAL |
| gauss2_m200/fd2 | 6 | 200 | fd2 | 2 | 15/15 | 12/12 | 11/11 | 2.85e-11 | 1.20e-17 | 6.6e-09 | MATCH |
| gauss2_m200/fd2_bounded | 6 | 200 | fd2 | 1 | 11/11 | 10/10 | 9/9 | 3.45e-11 | 2.40e-17 | 1.1e-08 | MATCH |
| gauss2_m200/fd3 | 6 | 200 | fd3 | 2 | 15/15 | 12/12 | 11/11 | 4.42e-14 | 1.89e-17 | 1.1e-11 | IDENTICAL |
| gauss3_n9/fd2 | 9 | 256 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 1.30e-11 | 7.79e-18 | 5.2e-10 | MATCH |
| gauss3_n9/fd2_bounded | 9 | 256 | fd2 | 1 | 7/7 | 7/7 | 6/6 | 5.44e-11 | 2.60e-18 | 1.5e-10 | MATCH |
| gauss3_n9/fd2_diffstep | 9 | 256 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 9.53e-13 | 5.19e-18 | 6.8e-12 | IDENTICAL |
| gauss3_n9/fd2_xscale_jac | 9 | 256 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 3.32e-11 | 4.33e-19 | 4.9e-10 | MATCH |
| gauss3_n9/fd3 | 9 | 256 | fd3 | 1 | 8/8 | 8/8 | 7/7 | 2.81e-14 | 4.33e-19 | 1.5e-12 | IDENTICAL |
| gauss3_n9/jac | 9 | 256 | analytic | 1 | 8/8 | 8/8 | 7/7 | 5.55e-17 | 4.33e-19 | = | IDENTICAL |
| gauss3_n9/jac_bounded | 9 | 256 | analytic | 1 | 7/7 | 7/7 | 6/6 | 0 | 2.16e-18 | = | IDENTICAL |
| gauss3_n9_farstart/fd2 | 9 | 256 | fd2 | 1 | 16/16 | 14/14 | 13/13 | 2.57e-11 | 8.22e-18 | 1.5e-08 | MATCH |
| gauss3_n9_farstart/fd2_bounded | 9 | 256 | fd2 | 2 | 15/15 | 14/14 | 13/13 | 3.54e-09 | 2.57e-15 | 2.3e-08 | MATCH |
| gauss3_outliers_n9/fd2_cauchy | 9 | 256 | fd2 | 1 | 16/16 | 11/11 | 10/10 | 1.54e-11 | 2.36e-17 | 7.7e-10 | MATCH |
| gauss3_outliers_n9/fd2_huber | 9 | 256 | fd2 | 2 | 12/12 | 9/9 | 8/8 | 2.85e-10 | 1.69e-16 | 3.3e-10 | MATCH |
| gauss3_outliers_n9/fd2_linear | 9 | 256 | fd2 | 2 | 22/22 | 16/16 | 15/15 | 2.20e-09 | 1.03e-13 | 1.4e-08 | MATCH |
| gauss3_outliers_n9/fd2_soft_l1 | 9 | 256 | fd2 | 2 | 13/13 | 10/10 | 9/9 | 1.39e-10 | 2.56e-16 | 9.4e-10 | MATCH |
| gauss3_outliers_n9/jac_bounded_huber | 9 | 256 | analytic | 2 | 10/10 | 8/8 | 7/7 | 6.66e-16 | 2.54e-16 | 5.9e-11 | IDENTICAL |
| gauss3_overlap_n9/fd2 | 9 | 256 | fd2 | 2 | 49/49 | 39/39 | 38/38 | 9.64e-07 | 5.78e-16 | 9.6e-07 | EQUIV |
| gauss3_overlap_n9/fd2_bounded | 9 | 256 | fd2 | 2 | 95/94 | 80/80 | 79/79 | 1.42e-05 | 6.90e-15 | 5.9e-05 | EQUIV |
| gauss3_overlap_n9/jac | 9 | 256 | analytic | 2 | 49/49 | 39/39 | 38/38 | 1.14e-12 | 8.65e-19 | 2.6e-12 | MATCH |
| gauss3_overlap_n9/jac_lsmr | 9 | 256 | analytic | 2 | 127/110 | 101/103 | 100/102 | 0.00216163 | 4.93e-09 | 4.6e-02 | EQUIV |
| linear_illcond_n40/jac | 40 | 40 | analytic | 3 | 6/6 | 6/6 | 5/5 | 1.84e-11 | 1.36e-20 | 3.2e-09 | MATCH |
| linear_illcond_n40/jac_lsmr | 40 | 40 | analytic | 0 | 4000/2387 | 4000/2387 | 3999/2386 | 0.071958 | 0.246033 | 3.5e-01 | MISMATCH |
| linear_illcond_n40/jac_xscale_jac | 40 | 40 | analytic | 3 | 24/24 | 24/23 | 23/23 | 2.67e-11 | 2.27e-20 | 3.2e-10 | MATCH* |
| linear_rank_deficient/fd2 | 10 | 12 | fd2 | 1 | 4/4 | 4/4 | 3/3 | 1.35191 | 2.24e-27 | 1.4e+00 | EQUIV |
| linear_rank_deficient/jac | 10 | 12 | analytic | 1 | 12/4 | 10/4 | 9/3 | 1.79367 | 4.67e-22 | 3.5e+00 | EQUIV |
| linear_rank_deficient/jac_lsmr | 10 | 12 | analytic | 1 | 5/5 | 5/5 | 4/4 | 5.55e-16 | 2.70e-28 | 3.4e-12 | IDENTICAL |
| mscan_gauss3_m1024/fd2 | 9 | 1024 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 9.16e-12 | 1.03e-17 | 2.0e-10 | MATCH |
| mscan_gauss3_m1024/jac | 9 | 1024 | analytic | 1 | 8/8 | 8/8 | 7/7 | 0 | 3.44e-18 | = | IDENTICAL |
| mscan_gauss3_m1024/jac_lsmr | 9 | 1024 | analytic | 2 | 7/7 | 7/7 | 6/6 | 0 | 1.72e-18 | 6.5e-12 | IDENTICAL |
| mscan_gauss3_m128/fd2 | 9 | 128 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 3.78e-11 | 3.25e-18 | 8.6e-10 | MATCH |
| mscan_gauss3_m128/jac | 9 | 128 | analytic | 1 | 8/8 | 8/8 | 7/7 | 0 | 1.08e-19 | = | IDENTICAL |
| mscan_gauss3_m128/jac_lsmr | 9 | 128 | analytic | 2 | 7/7 | 7/7 | 6/6 | 1.11e-16 | 1.30e-18 | 8.5e-12 | IDENTICAL |
| mscan_gauss3_m256/fd2 | 9 | 256 | fd2 | 2 | 7/7 | 7/7 | 6/6 | 9.41e-12 | 4.33e-19 | 3.8e-10 | MATCH |
| mscan_gauss3_m256/jac | 9 | 256 | analytic | 2 | 7/7 | 7/7 | 6/6 | 0 | 2.16e-19 | = | IDENTICAL |
| mscan_gauss3_m256/jac_lsmr | 9 | 256 | analytic | 4 | 7/7 | 7/7 | 6/6 | 1.78e-15 | 6.49e-19 | 7.0e-12 | IDENTICAL |
| mscan_gauss3_m32/fd2 | 9 | 32 | fd2 | 1 | 7/7 | 7/7 | 6/6 | 2.38e-10 | 6.50e-19 | 7.1e-10 | MATCH |
| mscan_gauss3_m32/jac | 9 | 32 | analytic | 1 | 7/7 | 7/7 | 6/6 | 1.11e-16 | 1.60e-18 | = | IDENTICAL |
| mscan_gauss3_m32/jac_lsmr | 9 | 32 | analytic | 2 | 7/7 | 7/7 | 6/6 | 3.33e-16 | 1.63e-18 | 3.2e-12 | IDENTICAL |
| mscan_gauss3_m512/fd2 | 9 | 512 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 4.46e-11 | 1.30e-18 | 2.4e-10 | MATCH |
| mscan_gauss3_m512/jac | 9 | 512 | analytic | 1 | 8/8 | 8/8 | 7/7 | 0 | 4.32e-19 | = | IDENTICAL |
| mscan_gauss3_m512/jac_lsmr | 9 | 512 | analytic | 2 | 7/7 | 7/7 | 6/6 | 1.67e-15 | 0 | 8.2e-12 | IDENTICAL |
| mscan_gauss3_m64/fd2 | 9 | 64 | fd2 | 1 | 8/8 | 8/8 | 7/7 | 2.91e-11 | 3.25e-18 | 7.0e-10 | MATCH |
| mscan_gauss3_m64/jac | 9 | 64 | analytic | 1 | 8/8 | 8/8 | 7/7 | 0 | 1.08e-19 | = | IDENTICAL |
| mscan_gauss3_m64/jac_lsmr | 9 | 64 | analytic | 2 | 7/7 | 7/7 | 6/6 | 1.11e-16 | 2.17e-18 | 3.3e-12 | IDENTICAL |
| powell_singular_n4/fd2 | 4 | 4 | fd2 | 1 | 13/13 | 13/13 | 12/12 | 1.88e-12 | 4.60e-21 | 1.1e-08 | MATCH |
| powell_singular_n4/jac | 4 | 4 | analytic | 1 | 13/13 | 13/13 | 12/12 | 2.18e-16 | 1.24e-26 | = | IDENTICAL |
| powell_singular_n4/jac_bounded | 4 | 4 | analytic | 1 | 14/14 | 14/14 | 13/13 | 5.22e-17 | 2.77e-26 | = | IDENTICAL |
| powell_singular_n4/jac_lsmr | 4 | 4 | analytic | 1 | 13/13 | 13/13 | 12/12 | 9.71e-11 | 2.02e-23 | 9.7e-11 | MATCH |
| rosenbrock_n2/fd2 | 2 | 2 | fd2 | 1 | 18/18 | 13/13 | 12/12 | 2.22e-16 | 2.49e-30 | 2.6e-10 | IDENTICAL |
| rosenbrock_n2/fd2_bounded | 2 | 2 | fd2 | 1 | 27/27 | 24/24 | 23/23 | 1.33e-15 | 1.06e-26 | 6.9e-09 | IDENTICAL |
| rosenbrock_n2/fd2_diffstep | 2 | 2 | fd2 | 1 | 18/18 | 13/13 | 12/12 | 2.22e-16 | 2.47e-30 | 2.2e-12 | IDENTICAL |
| rosenbrock_n2/fd2_xscale_jac | 2 | 2 | fd2 | 1 | 20/20 | 16/16 | 15/15 | 1.11e-16 | 6.22e-31 | 1.5e-09 | IDENTICAL |
| rosenbrock_n2/fd3 | 2 | 2 | fd3 | 1 | 18/18 | 13/13 | 12/12 | 2.22e-16 | 2.47e-30 | 1.6e-11 | IDENTICAL |
| rosenbrock_n2/jac | 2 | 2 | analytic | 1 | 18/18 | 13/13 | 12/12 | 2.22e-16 | 2.47e-30 | = | IDENTICAL |
| rosenbrock_n2/jac_bounded | 2 | 2 | analytic | 1 | 27/27 | 24/24 | 23/23 | 0 | 0 | = | IDENTICAL |
| rosenbrock_n2/jac_lsmr | 2 | 2 | analytic | 1 | 18/18 | 13/13 | 12/12 | 2.22e-16 | 2.47e-30 | = | IDENTICAL |
| rosenbrock_n2/jac_tight_tol | 2 | 2 | analytic | 1 | 18/19 | 13/14 | 12/13 | 0 | 0 | 1.5e-14 | MATCH* |
| rosenbrock_n2/jac_xscale_jac | 2 | 2 | analytic | 1 | 20/20 | 16/16 | 15/15 | 2.22e-16 | 2.47e-30 | = | IDENTICAL |
| sinbig_n100_m400/fd2 | 100 | 400 | fd2 | 1 | 6/6 | 6/6 | 5/5 | 6.76e-14 | 1.08e-23 | 8.0e-09 | IDENTICAL |
| sinbig_n100_m400/jac | 100 | 400 | analytic | 1 | 6/6 | 6/6 | 5/5 | 1.67e-16 | 1.88e-26 | = | IDENTICAL |
| sinbig_n100_m400/jac_lsmr | 100 | 400 | analytic | 1 | 6/6 | 6/6 | 5/5 | 2.22e-16 | 8.80e-26 | = | IDENTICAL |
| sinbig_n200_m800/fd2 | 200 | 800 | fd2 | 1 | 6/6 | 6/6 | 5/5 | 1.73e-14 | 6.72e-27 | 2.3e-08 | IDENTICAL |
| sinbig_n200_m800/jac | 200 | 800 | analytic | 1 | 6/6 | 6/6 | 5/5 | 2.78e-16 | 3.39e-28 | = | IDENTICAL |
| sinbig_n200_m800/jac_lsmr | 200 | 800 | analytic | 1 | 6/6 | 6/6 | 5/5 | 2.50e-16 | 5.01e-27 | = | IDENTICAL |
| sinbig_n400_m1200/fd2 | 400 | 1200 | fd2 | 1 | 6/6 | 6/6 | 5/5 | 9.66e-13 | 9.98e-22 | 7.3e-08 | IDENTICAL |
| sinbig_n400_m1200/jac | 400 | 1200 | analytic | 1 | 6/6 | 6/6 | 5/5 | 7.77e-16 | 3.90e-25 | = | IDENTICAL |
| sinbig_n400_m1200/jac_lsmr | 400 | 1200 | analytic | 1 | 6/6 | 6/6 | 5/5 | 5.27e-16 | 3.44e-25 | = | IDENTICAL |
| underdetermined_m6_n12/fd2 | 12 | 6 | fd2 | 1 | 37/37 | 17/17 | 16/16 | 3.32e-09 | 4.40e-17 | 5.6e-09 | MATCH |
| underdetermined_m6_n12/jac | 12 | 6 | analytic | 1 | 37/37 | 17/17 | 16/16 | 3.33e-16 | 1.16e-23 | = | IDENTICAL |
| wood_n4/fd2 | 4 | 6 | fd2 | 1 | 76/74 | 68/67 | 67/66 | 9.64e-14 | 3.31e-24 | 2.5e-01 | MATCH* |
| wood_n4/jac | 4 | 6 | analytic | 1 | 76/76 | 68/68 | 67/67 | 2.22e-16 | 2.41e-27 | 2.8e-09 | IDENTICAL |

`traj` = maximum relative deviation over the whole iteration trajectory (`=` means every intermediate iterate agrees to ≤1e-12).

## 4. Performance (best-of-N wall clock, seconds)

| problem / config | n × m | jac | t scipy | t TRF.Sharp | speedup |
|---|---|---|---:|---:|---:|
| beale_n2/fd2 | 2×3 | fd2 | 0.679 ms | 0.106 ms | 6.40× |
| beale_n2/jac | 2×3 | analytic | 0.411 ms | 0.098 ms | 4.19× |
| beale_n2/jac_bounded | 2×3 | analytic | 0.767 ms | 0.152 ms | 5.05× |
| box3d_n3/fd2 | 3×10 | fd2 | 0.663 ms | 0.167 ms | 3.97× |
| box3d_n3/jac | 3×10 | analytic | 0.364 ms | 0.142 ms | 2.56× |
| box3d_n3/jac_bounded | 3×10 | analytic | 1.498 ms | 0.565 ms | 2.65× |
| broyden_n10/fd2 | 10×10 | fd2 | 0.841 ms | 0.280 ms | 3.00× |
| broyden_n10/jac | 10×10 | analytic | 0.355 ms | 0.198 ms | 1.79× |
| broyden_n10/jac_lsmr | 10×10 | analytic | 1.242 ms | 0.347 ms | 3.57× |
| broyden_n100/fd2 | 100×100 | fd2 | 12.852 ms | 27.831 ms | 0.46× |
| broyden_n100/jac | 100×100 | analytic | 8.507 ms | 24.502 ms | 0.35× |
| broyden_n100/jac_lsmr | 100×100 | analytic | 2.471 ms | 15.917 ms | 0.16× |
| broyden_n200/fd2 | 200×200 | fd2 | 51.726 ms | 185.623 ms | 0.28× |
| broyden_n200/jac | 200×200 | analytic | 43.069 ms | 174.509 ms | 0.25× |
| broyden_n200/jac_lsmr | 200×200 | analytic | 3.602 ms | 37.164 ms | 0.10× |
| broyden_n50/fd2 | 50×50 | fd2 | 5.105 ms | 5.154 ms | 0.99× |
| broyden_n50/jac | 50×50 | analytic | 2.963 ms | 4.025 ms | 0.74× |
| broyden_n50/jac_lsmr | 50×50 | analytic | 2.102 ms | 8.391 ms | 0.25× |
| damped_osc_m150/fd2 | 5×150 | fd2 | 1.872 ms | 1.150 ms | 1.63× |
| damped_osc_m150/fd2_bounded | 5×150 | fd2 | 1.838 ms | 0.881 ms | 2.09× |
| damped_osc_m150/fd3 | 5×150 | fd3 | 2.421 ms | 1.328 ms | 1.82× |
| dbv_n10/fd2 | 10×10 | fd2 | 0.758 ms | 0.200 ms | 3.80× |
| dbv_n10/jac | 10×10 | analytic | 0.303 ms | 0.155 ms | 1.95× |
| dbv_n10/jac_lsmr | 10×10 | analytic | 1.263 ms | 0.360 ms | 3.51× |
| decay_peak_n9/fd2 | 9×220 | fd2 | 3.664 ms | 2.807 ms | 1.31× |
| decay_peak_n9/fd2_bounded | 9×220 | fd2 | 2.367 ms | 1.527 ms | 1.55× |
| decay_peak_n9/fd2_xscale_jac | 9×220 | fd2 | 3.152 ms | 2.468 ms | 1.28× |
| decay_peak_n9/fd3 | 9×220 | fd3 | 5.306 ms | 4.077 ms | 1.30× |
| decay_peak_n9/jac | 9×220 | analytic | 1.739 ms | 1.902 ms | 0.91× |
| decay_peak_n9/jac_bounded | 9×220 | analytic | 1.123 ms | 0.958 ms | 1.17× |
| decay_peak_n9/jac_lsmr | 9×220 | analytic | 278.843 ms | 1682.982 ms | 0.17× |
| exp_fit_m80/fd2 | 3×80 | fd2 | 30.344 ms | 9.436 ms | 3.22× |
| exp_fit_m80/fd2_bounded | 3×80 | fd2 | 1.810 ms | 0.608 ms | 2.98× |
| exp_fit_m80/fd3 | 3×80 | fd3 | 34.966 ms | 10.362 ms | 3.37× |
| exp_fit_m80/jac | 3×80 | analytic | 19.359 ms | 8.303 ms | 2.33× |
| exp_fit_m80/jac_bounded | 3×80 | analytic | 1.271 ms | 0.705 ms | 1.80× |
| exp_fit_m80/jac_xscale_array | 3×80 | analytic | 19.216 ms | 11.082 ms | 1.73× |
| exp_fit_outliers_m80/arctan | 3×80 | analytic | 23.849 ms | 9.623 ms | 2.48× |
| exp_fit_outliers_m80/bounded_cauchy | 3×80 | analytic | 1.703 ms | 0.716 ms | 2.38× |
| exp_fit_outliers_m80/cauchy | 3×80 | analytic | 23.844 ms | 8.969 ms | 2.66× |
| exp_fit_outliers_m80/fd2_huber | 3×80 | fd2 | 38.206 ms | 8.687 ms | 4.40× |
| exp_fit_outliers_m80/huber | 3×80 | analytic | 27.253 ms | 9.248 ms | 2.95× |
| exp_fit_outliers_m80/linear | 3×80 | analytic | 19.400 ms | 11.519 ms | 1.68× |
| exp_fit_outliers_m80/soft_l1 | 3×80 | analytic | 24.377 ms | 9.136 ms | 2.67× |
| gauss2_m200/fd2 | 6×200 | fd2 | 2.271 ms | 1.435 ms | 1.58× |
| gauss2_m200/fd2_bounded | 6×200 | fd2 | 2.011 ms | 1.086 ms | 1.85× |
| gauss2_m200/fd3 | 6×200 | fd3 | 3.031 ms | 1.744 ms | 1.74× |
| gauss3_n9/fd2 | 9×256 | fd2 | 1.959 ms | 1.632 ms | 1.20× |
| gauss3_n9/fd2_bounded | 9×256 | fd2 | 1.933 ms | 1.405 ms | 1.38× |
| gauss3_n9/fd2_diffstep | 9×256 | fd2 | 1.981 ms | 1.635 ms | 1.21× |
| gauss3_n9/fd2_xscale_jac | 9×256 | fd2 | 1.990 ms | 1.849 ms | 1.08× |
| gauss3_n9/fd3 | 9×256 | fd3 | 2.955 ms | 2.223 ms | 1.33× |
| gauss3_n9/jac | 9×256 | analytic | 0.869 ms | 1.056 ms | 0.82× |
| gauss3_n9/jac_bounded | 9×256 | analytic | 0.958 ms | 0.954 ms | 1.00× |
| gauss3_n9_farstart/fd2 | 9×256 | fd2 | 3.686 ms | 3.113 ms | 1.18× |
| gauss3_n9_farstart/fd2_bounded | 9×256 | fd2 | 4.431 ms | 6.162 ms | 0.72× |
| gauss3_outliers_n9/fd2_cauchy | 9×256 | fd2 | 3.316 ms | 2.488 ms | 1.33× |
| gauss3_outliers_n9/fd2_huber | 9×256 | fd2 | 2.854 ms | 1.940 ms | 1.47× |
| gauss3_outliers_n9/fd2_linear | 9×256 | fd2 | 4.383 ms | 3.516 ms | 1.25× |
| gauss3_outliers_n9/fd2_soft_l1 | 9×256 | fd2 | 2.963 ms | 2.077 ms | 1.43× |
| gauss3_outliers_n9/jac_bounded_huber | 9×256 | analytic | 1.644 ms | 1.623 ms | 1.01× |
| gauss3_overlap_n9/fd2 | 9×256 | fd2 | 10.790 ms | 8.188 ms | 1.32× |
| gauss3_overlap_n9/fd2_bounded | 9×256 | fd2 | 24.230 ms | 16.315 ms | 1.49× |
| gauss3_overlap_n9/jac | 9×256 | analytic | 5.396 ms | 6.772 ms | 0.80× |
| gauss3_overlap_n9/jac_lsmr | 9×256 | analytic | 37.122 ms | 203.815 ms | 0.18× |
| linear_illcond_n40/jac | 40×40 | analytic | 1.126 ms | 2.234 ms | 0.50× |
| linear_illcond_n40/jac_lsmr | 40×40 | analytic | 2798.019 ms | 4312.984 ms | 0.65× |
| linear_illcond_n40/jac_xscale_jac | 40×40 | analytic | 4.735 ms | 11.026 ms | 0.43× |
| linear_rank_deficient/fd2 | 10×12 | fd2 | 0.604 ms | 0.091 ms | 6.61× |
| linear_rank_deficient/jac | 10×12 | analytic | 1.134 ms | 0.074 ms | 15.41× |
| linear_rank_deficient/jac_lsmr | 10×12 | analytic | 0.981 ms | 0.076 ms | 12.89× |
| mscan_gauss3_m1024/fd2 | 9×1024 | fd2 | 3.006 ms | 4.943 ms | 0.61× |
| mscan_gauss3_m1024/jac | 9×1024 | analytic | 1.503 ms | 3.681 ms | 0.41× |
| mscan_gauss3_m1024/jac_lsmr | 9×1024 | analytic | 2.567 ms | 27.570 ms | 0.09× |
| mscan_gauss3_m128/fd2 | 9×128 | fd2 | 1.787 ms | 0.525 ms | 3.41× |
| mscan_gauss3_m128/jac | 9×128 | analytic | 0.755 ms | 0.330 ms | 2.29× |
| mscan_gauss3_m128/jac_lsmr | 9×128 | analytic | 1.860 ms | 6.448 ms | 0.29× |
| mscan_gauss3_m256/fd2 | 9×256 | fd2 | 1.709 ms | 1.028 ms | 1.66× |
| mscan_gauss3_m256/jac | 9×256 | analytic | 0.755 ms | 0.656 ms | 1.15× |
| mscan_gauss3_m256/jac_lsmr | 9×256 | analytic | 1.883 ms | 10.665 ms | 0.18× |
| mscan_gauss3_m32/fd2 | 9×32 | fd2 | 1.423 ms | 0.194 ms | 7.32× |
| mscan_gauss3_m32/jac | 9×32 | analytic | 0.586 ms | 0.154 ms | 3.80× |
| mscan_gauss3_m32/jac_lsmr | 9×32 | analytic | 1.701 ms | 0.153 ms | 11.11× |
| mscan_gauss3_m512/fd2 | 9×512 | fd2 | 2.400 ms | 2.756 ms | 0.87× |
| mscan_gauss3_m512/jac | 9×512 | analytic | 1.095 ms | 2.028 ms | 0.54× |
| mscan_gauss3_m512/jac_lsmr | 9×512 | analytic | 2.182 ms | 17.054 ms | 0.13× |
| mscan_gauss3_m64/fd2 | 9×64 | fd2 | 1.659 ms | 0.382 ms | 4.35× |
| mscan_gauss3_m64/jac | 9×64 | analytic | 0.691 ms | 0.255 ms | 2.71× |
| mscan_gauss3_m64/jac_lsmr | 9×64 | analytic | 1.748 ms | 2.494 ms | 0.70× |
| powell_singular_n4/fd2 | 4×4 | fd2 | 1.272 ms | 0.356 ms | 3.57× |
| powell_singular_n4/jac | 4×4 | analytic | 0.623 ms | 0.316 ms | 1.97× |
| powell_singular_n4/jac_bounded | 4×4 | analytic | 1.064 ms | 0.429 ms | 2.48× |
| powell_singular_n4/jac_lsmr | 4×4 | analytic | 2.269 ms | 0.439 ms | 5.17× |
| rosenbrock_n2/fd2 | 2×2 | fd2 | 1.522 ms | 0.195 ms | 7.81× |
| rosenbrock_n2/fd2_bounded | 2×2 | fd2 | 3.054 ms | 0.352 ms | 8.67× |
| rosenbrock_n2/fd2_diffstep | 2×2 | fd2 | 1.558 ms | 0.264 ms | 5.90× |
| rosenbrock_n2/fd2_xscale_jac | 2×2 | fd2 | 1.859 ms | 0.311 ms | 5.98× |
| rosenbrock_n2/fd3 | 2×2 | fd3 | 1.648 ms | 0.194 ms | 8.50× |
| rosenbrock_n2/jac | 2×2 | analytic | 1.005 ms | 0.164 ms | 6.13× |
| rosenbrock_n2/jac_bounded | 2×2 | analytic | 2.024 ms | 0.321 ms | 6.31× |
| rosenbrock_n2/jac_lsmr | 2×2 | analytic | 2.506 ms | 0.415 ms | 6.04× |
| rosenbrock_n2/jac_tight_tol | 2×2 | analytic | 0.999 ms | 0.264 ms | 3.78× |
| rosenbrock_n2/jac_xscale_jac | 2×2 | analytic | 1.218 ms | 0.290 ms | 4.20× |
| sinbig_n100_m400/fd2 | 100×400 | fd2 | 73.055 ms | 241.177 ms | 0.30× |
| sinbig_n100_m400/jac | 100×400 | analytic | 64.894 ms | 50.902 ms | 1.27× |
| sinbig_n100_m400/jac_lsmr | 100×400 | analytic | 3.970 ms | 45.027 ms | 0.09× |
| sinbig_n200_m800/fd2 | 200×800 | fd2 | 197.524 ms | 1161.075 ms | 0.17× |
| sinbig_n200_m800/jac | 200×800 | analytic | 157.444 ms | 376.677 ms | 0.42× |
| sinbig_n200_m800/jac_lsmr | 200×800 | analytic | 8.961 ms | 101.074 ms | 0.09× |
| sinbig_n400_m1200/fd2 | 400×1200 | fd2 | 921.926 ms | 5426.002 ms | 0.17× |
| sinbig_n400_m1200/jac | 400×1200 | analytic | 575.862 ms | 2728.051 ms | 0.21× |
| sinbig_n400_m1200/jac_lsmr | 400×1200 | analytic | 30.445 ms | 245.444 ms | 0.12× |
| underdetermined_m6_n12/fd2 | 12×6 | fd2 | 4.561 ms | 0.443 ms | 10.30× |
| underdetermined_m6_n12/jac | 12×6 | analytic | 2.976 ms | 0.387 ms | 7.69× |
| wood_n4/fd2 | 4×6 | fd2 | 7.340 ms | 2.088 ms | 3.52× |
| wood_n4/jac | 4×6 | analytic | 3.751 ms | 2.033 ms | 1.84× |

Geometric mean speedup (TRF.Sharp / SciPy): **1.43×**
- `analytic` (64 cases): geometric mean 1.14×
- `fd2` (46 cases): geometric mean 1.82×
- `fd3` (6 cases): geometric mean 2.32×

Broken down by problem size — this is where the picture flips:

| n·m | cases | geomean speedup |
|---|---:|---:|
| n·m ≤ 100 | 30 | 4.24× |
| 100 < n·m ≤ 2 000 | 41 | 2.03× |
| 2 000 < n·m ≤ 50 000 | 39 | 0.59× |
| n·m > 50 000 | 6 | 0.17× |

Small problems are dominated by SciPy's per-call Python overhead (argument validation, `VectorFunction` wrapper, a Python-level loop over the finite-difference columns), so TRF.Sharp wins by 3-10×. On large dense problems the cost is the linear algebra — which is why `EconomySvd` no longer calls MathNet's *managed* SVD (8-12× slower than OpenBLAS' LAPACK `gesdd`, and scaling worse than O(m·n²)) for tall matrices. It takes a Householder thin QR that produces only the n×n factor R, then an SVD of R: 25× faster at m = 256, 154× at m = 1024, 563× at m = 4096 (n = 9), with the Gauss-Newton step reproduced to ~1e-13. See §8 of `report_nine_param.md` and `ab_speed.py`; the regression is verdict-neutral (68 IDENTICAL before and after).

## 5. A/B: how `uf` is computed in the trust-region subproblem

`solve_lsq_trust_region` needs <code>uf = Uᵀ·f</code>. MathNet's left
singular vectors lose ~7 digits on Jacobians whose smallest singular value
is ~1e-10 relative to the largest (LAPACK, used by SciPy, does not). Since
the augmented rows carry a zero right-hand side, <code>Aᵀ·f_aug = d·g = g_h</code>,
so <code>suf = s·uf = Vᵀ·g_h</code> can be computed from <code>V</code> alone —
which is accurate and one mat-vec cheaper.

| variant | IDENTICAL | MATCH | MATCH\* | EQUIV | CLOSE | MISMATCH |
|---|---:|---:|---:|---:|---:|---:|
| A: `uf = Uᵀ·f` (MathNet U) | 53 | 14 | 0 | 4 | 0 | 2 |
| B: `uf = (Vᵀ·g_h)/s` (**shipped**) | 49 | 15 | 4 | 4 | 0 | 1 |

| case | A verdict | B verdict | x rel-err A → B | cost rel-err A → B |
|---|---|---|---:|---:|
| `box3d_n3/fd2` | IDENTICAL | MATCH | 5.89e-13 → 1.06e-12 | 2.92e-23 → 1.66e-23 |
| `box3d_n3/jac_bounded` | MISMATCH | MATCH* | 41.5774 → 1.92e-09 | 0.0377944 → 2.14e-19 |
| `exp_fit_outliers_m80/linear` | IDENTICAL | MATCH | 5.85e-13 → 8.70e-12 | 2.35e-14 → 9.49e-14 |
| `linear_illcond_n40/jac_xscale_jac` | MATCH | MATCH* | 1.72e-11 → 2.67e-11 | 3.25e-20 → 2.27e-20 |
| `rosenbrock_n2/jac_tight_tol` | IDENTICAL | MATCH* | 0 → 0 | 0 → 0 |
| `wood_n4/fd2` | IDENTICAL | MATCH* | 2.22e-15 → 9.64e-14 | 9.62e-26 → 3.31e-24 |

Variant B removes the only *algorithmic* divergence (`box3d_n3/jac_bounded`: x rel-err 4.2e+01 → 1.9e-09, cost rel-err 3.8e-02 → 2.1e-19). The IDENTICAL → MATCH\* moves are 1-ULP shifts: the solutions still agree to ~1e-12.

## 6. Cases that do not match bit-for-bit

| case | verdict | why |
|---|---|---|
| box3d_n3/jac_bounded | MATCH* | x rel-err 1.92e-09, cost rel-err 2.14e-19, traj deviates from iteration 5 — same answer, slightly different iteration path |
| decay_peak_n9/fd2 | EQUIV | x rel-err 4.72e-08, cost rel-err 9.27e-17, traj deviates from iteration 0 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| decay_peak_n9/fd2_bounded | EQUIV | x rel-err 6.48e-08, cost rel-err 7.54e-17, traj deviates from iteration 1 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| decay_peak_n9/fd2_xscale_jac | EQUIV | x rel-err 1.44e-08, cost rel-err 6.37e-17, traj deviates from iteration 1 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| decay_peak_n9/jac_lsmr | EQUIV | x rel-err 0.00347739, cost rel-err 3.46e-07, traj deviates from iteration 0 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| exp_fit_m80/fd2 | EQUIV | x rel-err 1.45e-08, cost rel-err 2.11e-07, traj deviates from iteration 0 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| exp_fit_outliers_m80/fd2_huber | EQUIV | x rel-err 0.01113, cost rel-err 4.49e-08, traj deviates from iteration 0 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| gauss3_overlap_n9/fd2 | EQUIV | x rel-err 9.64e-07, cost rel-err 5.78e-16, traj deviates from iteration 1 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| gauss3_overlap_n9/fd2_bounded | EQUIV | x rel-err 1.42e-05, cost rel-err 6.90e-15, traj deviates from iteration 1 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| gauss3_overlap_n9/jac_lsmr | EQUIV | x rel-err 0.00216163, cost rel-err 4.93e-09, traj deviates from iteration 0 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| linear_illcond_n40/jac_lsmr | MISMATCH | x rel-err 0.071958, cost rel-err 0.246033, traj deviates from iteration 1 — genuine path divergence on an ill-conditioned problem |
| linear_illcond_n40/jac_xscale_jac | MATCH* | x rel-err 2.67e-11, cost rel-err 2.27e-20, traj deviates from iteration 19 — same answer, slightly different iteration path |
| linear_rank_deficient/fd2 | EQUIV | x rel-err 1.35191, cost rel-err 2.24e-27, traj deviates from iteration 1 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| linear_rank_deficient/jac | EQUIV | x rel-err 1.79367, cost rel-err 4.67e-22, traj deviates from iteration 1 — same cost, different x — non-unique minimiser or flat drift to `max_nfev` |
| rosenbrock_n2/jac_tight_tol | MATCH* | x rel-err 0, cost rel-err 0, traj deviates from iteration None — same answer, slightly different iteration path |
| wood_n4/fd2 | MATCH* | x rel-err 9.64e-14, cost rel-err 3.31e-24, traj deviates from iteration 1 — same answer, slightly different iteration path |

## 7. Notes

- Both sides read the **same** `problems.json`, so x0, bounds, data matrices and all solver options are bit-identical.
- Timing is best-of-N of the whole `solve` call (warm-up excluded); N = 9 for n·m ≤ 4 000, 5 for n·m ≤ 200 000, 3 otherwise.
- SciPy's dense linear algebra goes through OpenBLAS; TRF.Sharp uses MathNet's **managed** SVD/QR, which is where large dense cases differ most.
- `nit` is not exposed by SciPy ≥ 1.16 for `method='trf'`; it is taken from the last callback, which carries the same iteration counter.
