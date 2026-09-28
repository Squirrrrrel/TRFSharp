"""Generate the shared benchmark problem set for TRF.Sharp vs SciPy.

The same ``problems.json`` is consumed by

    bench/csharp/BenchRunner   (TRF.Sharp, .NET)
    bench/run_python.py        (scipy.optimize.least_squares)

so both sides see bit-identical inputs.  Matrices are stored as
base64-encoded little-endian float64 so that the values round-trip exactly.
"""

from __future__ import annotations

import base64
import math
import json
import numpy as np

SEED = 20260928
OUT = "problems.json"


def b64f64(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return {
        "shape": list(a.shape),
        "b64": base64.b64encode(a.tobytes()).decode("ascii"),
    }


def cfg(
    cid: str,
    jac: str = "fd2",
    lb=None,
    ub=None,
    x_scale=None,
    loss=None,
    f_scale=None,
    tr_solver=None,
    diff_step=None,
    ftol=None,
    xtol=None,
    gtol=None,
) -> dict:
    def arr(v):
        return None if v is None else [float(t) for t in np.asarray(v, float).ravel()]

    return {
        "id": cid,
        "jac": jac,
        "lb": arr(lb),
        "ub": arr(ub),
        "x_scale": x_scale,
        "loss": loss,
        "f_scale": f_scale,
        "tr_solver": tr_solver,
        "diff_step": diff_step,
        "ftol": ftol,
        "xtol": xtol,
        "gtol": gtol,
    }


problems: list[dict] = []


def add(name: str, n: int, m: int, x0, data=None, configs=()):
    problems.append(
        {
            "name": name,
            "n": int(n),
            "m": int(m),
            "x0": [float(v) for v in np.asarray(x0, dtype=float).ravel()],
            "data": data or {},
            "configs": list(configs),
        }
    )


rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------
# 1. Rosenbrock  n=2, m=2
# --------------------------------------------------------------------------
add("rosenbrock_n2", 2, 2, [-2.0, 1.0], configs=[
    cfg("jac", jac="analytic"),
    cfg("fd2", jac="fd2"),
    cfg("fd3", jac="fd3"),
    cfg("jac_bounded", jac="analytic", lb=[-np.inf, -1.5], ub=[np.inf, 1.5]),
    cfg("fd2_bounded", jac="fd2", lb=[-np.inf, -1.5], ub=[np.inf, 1.5]),
    cfg("jac_xscale_jac", jac="analytic", x_scale="jac"),
    cfg("fd2_xscale_jac", jac="fd2", x_scale="jac"),
    cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
    cfg("fd2_diffstep", jac="fd2", diff_step=1e-4),
    cfg("jac_tight_tol", jac="analytic", ftol=1e-14, xtol=1e-14, gtol=1e-14),
])

# --------------------------------------------------------------------------
# 2. Beale  n=2, m=3
# --------------------------------------------------------------------------
add("beale_n2", 2, 3, [1.0, 1.0], configs=[
    cfg("jac", jac="analytic"),
    cfg("fd2", jac="fd2"),
    cfg("jac_bounded", jac="analytic", lb=[0.0, -5.0], ub=[10.0, 5.0]),
])

# --------------------------------------------------------------------------
# 3. Powell singular  n=4, m=4  (singular Jacobian at the solution)
# --------------------------------------------------------------------------
add("powell_singular_n4", 4, 4, [3.0, -1.0, 0.0, 1.0], configs=[
    cfg("jac", jac="analytic"),
    cfg("fd2", jac="fd2"),
    cfg("jac_bounded", jac="analytic", lb=[-10.0] * 4, ub=[10.0] * 4),
    cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
])

# --------------------------------------------------------------------------
# 4. Wood function  n=4, m=6
# --------------------------------------------------------------------------
add("wood_n4", 4, 6, [-3.0, -1.0, -3.0, -1.0], configs=[
    cfg("jac", jac="analytic"),
    cfg("fd2", jac="fd2"),
])

# --------------------------------------------------------------------------
# 5. Box 3-dimensional  n=3, m=10
# --------------------------------------------------------------------------
add("box3d_n3", 3, 10, [0.0, 10.0, 20.0], configs=[
    cfg("jac", jac="analytic"),
    cfg("fd2", jac="fd2"),
    cfg("jac_bounded", jac="analytic", lb=[-1.0, 0.0, 0.0]),
])

# --------------------------------------------------------------------------
# 6. Discrete boundary value (scalable)  n=10, m=10
# --------------------------------------------------------------------------
add("dbv_n10", 10, 10, np.full(10, -1.0), configs=[
    cfg("jac", jac="analytic"),
    cfg("fd2", jac="fd2"),
    cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
])

# --------------------------------------------------------------------------
# 7. Broyden tridiagonal (scalable)
# --------------------------------------------------------------------------
for _n in (10, 50, 100, 200):
    add(f"broyden_n{_n}", _n, _n, np.full(_n, -1.0), configs=[
        cfg("jac", jac="analytic"),
        cfg("fd2", jac="fd2"),
        cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
    ])

# --------------------------------------------------------------------------
# 8. Rank-deficient linear system  m=12, n=10, rank=6
# --------------------------------------------------------------------------
_A = rng.standard_normal((12, 6)) @ rng.standard_normal((6, 10))
_b = _A @ rng.standard_normal(10)
add("linear_rank_deficient", 10, 12, np.zeros(10),
    data={"A": b64f64(_A), "b": b64f64(_b)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
        cfg("fd2", jac="fd2"),
    ])

# --------------------------------------------------------------------------
# 9. Underdetermined  m=6, n=12
# --------------------------------------------------------------------------
_A = rng.standard_normal((6, 12))
_b = _A @ rng.standard_normal(12)
add("underdetermined_m6_n12", 12, 6, np.zeros(12),
    data={"A": b64f64(_A), "b": b64f64(_b)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("fd2", jac="fd2"),
    ])

# --------------------------------------------------------------------------
# 10. Ill-conditioned linear system  m=n=40, cond ~ 1e6
# --------------------------------------------------------------------------
_U, _ = np.linalg.qr(rng.standard_normal((40, 40)))
_V, _ = np.linalg.qr(rng.standard_normal((40, 40)))
_s = np.logspace(0, 6, 40)
_A = (_U * _s) @ _V.T
_b = _A @ rng.standard_normal(40)
add("linear_illcond_n40", 40, 40, np.zeros(40),
    data={"A": b64f64(_A), "b": b64f64(_b)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("jac_xscale_jac", jac="analytic", x_scale="jac"),
        cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
    ])

# --------------------------------------------------------------------------
# 11. Exponential curve fit (data driven, seeded)  m=80, n=3
# --------------------------------------------------------------------------
_t = np.linspace(0.0, 3.0, 80)
_ptrue = np.array([2.5, -1.3, -0.8])
_y = _ptrue[0] + _ptrue[1] * np.exp(_ptrue[2] * _t) + 0.02 * rng.standard_normal(80)
add("exp_fit_m80", 3, 80, [1.0, 1.0, 0.0],
    data={"t": b64f64(_t), "y": b64f64(_y)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("fd2", jac="fd2"),
        cfg("fd3", jac="fd3"),
        cfg("jac_bounded", jac="analytic", lb=[0.0, -5.0, -5.0], ub=[5.0, 5.0, 5.0]),
        cfg("fd2_bounded", jac="fd2", lb=[0.0, -5.0, -5.0], ub=[5.0, 5.0, 5.0]),
        cfg("jac_xscale_array", jac="analytic", x_scale=[1.0, 2.0, 0.5]),
    ])

# --------------------------------------------------------------------------
# 12. Exponential fit with 15% outliers -> robust losses
# --------------------------------------------------------------------------
_y_out = _y.copy()
_idx = rng.choice(80, 12, replace=False)
_y_out[_idx] += rng.choice([-1.0, 1.0], 12) * rng.uniform(3.0, 8.0, 12)
add("exp_fit_outliers_m80", 3, 80, [1.0, 1.0, 0.0],
    data={"t": b64f64(_t), "y": b64f64(_y_out)}, configs=[
        cfg("linear", jac="analytic", loss="linear", f_scale=1.0),
        cfg("soft_l1", jac="analytic", loss="soft_l1", f_scale=0.5),
        cfg("huber", jac="analytic", loss="huber", f_scale=0.5),
        cfg("cauchy", jac="analytic", loss="cauchy", f_scale=0.5),
        cfg("arctan", jac="analytic", loss="arctan", f_scale=0.5),
        cfg("fd2_huber", jac="fd2", loss="huber", f_scale=0.5),
        cfg("bounded_cauchy", jac="analytic", loss="cauchy", f_scale=0.5,
            lb=[0.0, -5.0, -5.0], ub=[5.0, 5.0, 5.0]),
    ])

# --------------------------------------------------------------------------
# 13. Two-Gaussian peak fit  m=200, n=6  (classic local-minimum trap)
# --------------------------------------------------------------------------
_t2 = np.linspace(-3.0, 6.0, 200)
_p2 = np.array([3.0, 0.5, 0.7, 2.0, 3.0, 1.1])
_y2 = (_p2[0] * np.exp(-0.5 * ((_t2 - _p2[1]) / _p2[2]) ** 2)
       + _p2[3] * np.exp(-0.5 * ((_t2 - _p2[4]) / _p2[5]) ** 2)
       + 0.01 * rng.standard_normal(200))
add("gauss2_m200", 6, 200, [1.0, -0.5, 1.0, 1.0, 2.0, 1.0],
    data={"t": b64f64(_t2), "y": b64f64(_y2)}, configs=[
        cfg("fd2", jac="fd2"),
        cfg("fd3", jac="fd3"),
        cfg("fd2_bounded", jac="fd2",
            lb=[0.0, -3.0, 0.05, 0.0, 0.0, 0.05],
            ub=[10.0, 5.0, 5.0, 10.0, 6.0, 5.0]),
    ])

# --------------------------------------------------------------------------
# 14. Damped oscillation  m=150, n=5  (hard, many local minima)
# --------------------------------------------------------------------------
_t3 = np.linspace(0.0, 10.0, 150)
_p3 = np.array([1.5, 0.25, 2.0 * np.pi * 1.1, 0.4, 0.2])
_y3 = (_p3[0] * np.exp(-_p3[1] * _t3) * np.sin(_p3[2] * _t3 + _p3[3]) + _p3[4]
       + 0.01 * rng.standard_normal(150))
add("damped_osc_m150", 5, 150, [1.0, 0.1, 6.0, 0.0, 0.0],
    data={"t": b64f64(_t3), "y": b64f64(_y3)}, configs=[
        cfg("fd2", jac="fd2"),
        cfg("fd3", jac="fd3"),
        cfg("fd2_bounded", jac="fd2",
            lb=[0.0, 0.0, 0.0, -np.pi, -1.0],
            ub=[10.0, 5.0, 20.0, np.pi, 1.0]),
    ])

# --------------------------------------------------------------------------
# 15. Scalable nonlinear problem:  f_i = sum_j A_ij sin(x_j) - b_i
#     jac_ij = A_ij cos(x_j)
# --------------------------------------------------------------------------
for _n, _m in ((100, 400), (200, 800), (400, 1200)):
    _A = rng.standard_normal((_m, _n)) / np.sqrt(_n)
    _xs = rng.standard_normal(_n) * 0.3
    _b = _A @ np.sin(_xs)
    add(f"sinbig_n{_n}_m{_m}", _n, _m, np.full(_n, 0.1),
        data={"A": b64f64(_A), "b": b64f64(_b)},
        configs=[cfg("jac", jac="analytic"),
                 cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
                 cfg("fd2", jac="fd2")])

# --------------------------------------------------------------------------
# 16-19. NINE-PARAMETER nonlinear curve fits
#     The target workload: p = [A1, mu1, s1, A2, mu2, s2, A3, mu3, s3] or
#     [A1, tau1, A2, tau2, A3, tau3, c0, c1, c2].
# --------------------------------------------------------------------------

# 16. Three separated Gaussian peaks on a flat baseline -- well conditioned.
_t9 = np.linspace(-1.0, 9.0, 256)
_p9 = np.array([1.00, 1.50, 0.55, 0.65, 4.00, 0.90, 0.40, 6.80, 0.45])
_y9 = np.zeros_like(_t9)
for _k in range(3):
    _y9 = _y9 + _p9[3 * _k] * np.exp(
        -0.5 * ((_t9 - _p9[3 * _k + 1]) / _p9[3 * _k + 2]) ** 2)
_y9 = _y9 + 0.004 * rng.standard_normal(256)

_G_LB = [0.0, -2.0, 0.05, 0.0, 0.0, 0.05, 0.0, 4.0, 0.05]
_G_UB = [10.0, 4.0, 3.0, 10.0, 6.0, 3.0, 10.0, 10.0, 3.0]

add("gauss3_n9", 9, 256, [1.2, 1.2, 0.80, 0.90, 4.40, 0.80, 0.60, 6.40, 0.80],
    data={"t": b64f64(_t9), "y": b64f64(_y9)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("fd2", jac="fd2"),
        cfg("fd3", jac="fd3"),
        cfg("fd2_bounded", jac="fd2", lb=_G_LB, ub=_G_UB),
        cfg("jac_bounded", jac="analytic", lb=_G_LB, ub=_G_UB),
        cfg("fd2_xscale_jac", jac="fd2", x_scale="jac"),
        cfg("fd2_diffstep", jac="fd2", diff_step=1e-6),
    ])

# 17. Same data, deliberately poor starting point (fit must still converge).
add("gauss3_n9_farstart", 9, 256,
    [0.50, 0.00, 1.50, 0.50, 5.50, 1.50, 0.30, 4.50, 0.30],
    data={"t": b64f64(_t9), "y": b64f64(_y9)}, configs=[
        cfg("fd2", jac="fd2"),
        cfg("fd2_bounded", jac="fd2", lb=_G_LB, ub=_G_UB),
    ])

# 18. Two of the three peaks nearly coincide -> Jacobian is close to
#     rank deficient.  Stresses the trust-region subproblem.
_t9b = np.linspace(-1.0, 9.0, 256)
_p9b = np.array([1.00, 3.00, 0.80, 0.50, 5.00, 0.90, 0.45, 5.12, 0.85])
_y9b = np.zeros_like(_t9b)
for _k in range(3):
    _y9b = _y9b + _p9b[3 * _k] * np.exp(
        -0.5 * ((_t9b - _p9b[3 * _k + 1]) / _p9b[3 * _k + 2]) ** 2)
_y9b = _y9b + 0.004 * rng.standard_normal(256)
add("gauss3_overlap_n9", 9, 256,
    [1.10, 2.90, 0.90, 0.60, 4.80, 1.00, 0.55, 5.30, 0.90],
    data={"t": b64f64(_t9b), "y": b64f64(_y9b)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("fd2", jac="fd2"),
        cfg("fd2_bounded", jac="fd2", lb=_G_LB, ub=_G_UB),
        cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
    ])

# 19. Two-exponential decay background + one Gaussian peak + linear
#     baseline -- nine parameters, cond(J) ~ 1.4e4, unique minimiser.
#     (A pure 3-exponential 9-parameter fit is not identifiable: swapping the
#     noise realisation flips it between status 1 and max_nfev, so it cannot
#     serve as a reference case.)
_t9c = np.linspace(0.0, 20.0, 220)
_p9c = np.array([2.00, 0.80, 0.70, 6.00, 1.10, 7.50, 0.90, 0.12, -0.01])
_y9c = (_p9c[0] * np.exp(-_t9c / _p9c[1])
        + _p9c[2] * np.exp(-_t9c / _p9c[3])
        + _p9c[4] * np.exp(-0.5 * ((_t9c - _p9c[5]) / _p9c[6]) ** 2)
        + _p9c[7] + _p9c[8] * _t9c)
_y9c = _y9c + 0.003 * rng.standard_normal(220)
add("decay_peak_n9", 9, 220,
    [2.30, 0.60, 0.90, 5.00, 1.30, 7.00, 1.20, 0.10, 0.0],
    data={"t": b64f64(_t9c), "y": b64f64(_y9c)}, configs=[
        cfg("jac", jac="analytic"),
        cfg("fd2", jac="fd2"),
        cfg("fd3", jac="fd3"),
        cfg("fd2_bounded", jac="fd2",
            lb=[0.0, 0.02, 0.0, 0.02, 0.0, 0.0, 0.02, -1.0, -1.0],
            ub=[20.0, 100.0, 20.0, 100.0, 20.0, 20.0, 5.0, 1.0, 1.0]),
        cfg("jac_bounded", jac="analytic",
            lb=[0.0, 0.02, 0.0, 0.02, 0.0, 0.0, 0.02, -1.0, -1.0],
            ub=[20.0, 100.0, 20.0, 100.0, 20.0, 20.0, 5.0, 1.0, 1.0]),
        cfg("fd2_xscale_jac", jac="fd2", x_scale="jac"),
        cfg("jac_lsmr", jac="analytic", tr_solver="lsmr"),
    ])

# 20. Nine-parameter peak fit on data with 8% outliers -> robust losses.
_y9d = _y9.copy()
_idx9 = rng.choice(256, 20, replace=False)
_y9d[_idx9] += rng.choice([-1.0, 1.0], 20) * rng.uniform(0.8, 2.5, 20)
add("gauss3_outliers_n9", 9, 256,
    [1.2, 1.2, 0.80, 0.90, 4.40, 0.80, 0.60, 6.40, 0.80],
    data={"t": b64f64(_t9), "y": b64f64(_y9d)}, configs=[
        cfg("fd2_linear", jac="fd2", loss="linear", f_scale=1.0),
        cfg("fd2_soft_l1", jac="fd2", loss="soft_l1", f_scale=0.05),
        cfg("fd2_huber", jac="fd2", loss="huber", f_scale=0.05),
        cfg("fd2_cauchy", jac="fd2", loss="cauchy", f_scale=0.05),
        cfg("jac_bounded_huber", jac="analytic", loss="huber", f_scale=0.05,
            lb=_G_LB, ub=_G_UB),
    ])

# --------------------------------------------------------------------------
# 21. Data-count sweep: the SAME nine-parameter three-peak model sampled at
#     m = 32 ... 1024.  Isolates how the TRF.Sharp / SciPy speed ratio moves
#     with the size of the Jacobian (m x 9).
# --------------------------------------------------------------------------
for _m in (32, 64, 128, 256, 512, 1024):
    _t = np.linspace(-1.0, 9.0, _m)
    _y = np.zeros_like(_t)
    for _k in range(3):
        _y = _y + _p9[3 * _k] * np.exp(
            -0.5 * ((_t - _p9[3 * _k + 1]) / _p9[3 * _k + 2]) ** 2)
    _y = _y + 0.004 * rng.standard_normal(_m)
    add(f"mscan_gauss3_m{_m}", 9, _m,
        [1.2, 1.2, 0.80, 0.90, 4.40, 0.80, 0.60, 6.40, 0.80],
        data={"t": b64f64(_t), "y": b64f64(_y)},
        configs=[cfg("fd2", jac="fd2"),
                 cfg("jac", jac="analytic"),
                 cfg("jac_lsmr", jac="analytic", tr_solver="lsmr")])


def sanitize(o):
    """JSON has no +/-Infinity; emit them as the strings 'inf'/'-inf' so that
    both the Python and the .NET reader can restore them exactly."""
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, float):
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return o
    return o


doc = sanitize({
    "seed": SEED,
    "note": "Shared problem set. Matrices are base64 little-endian float64.",
    "problems": problems,
})

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(doc, fh)

n_cfg = sum(len(p["configs"]) for p in problems)
print(f"problems.json: {len(problems)} problems, {n_cfg} cases, "
      f"{len(json.dumps(doc)) / 1e6:.2f} MB")
for p in problems:
    print(f"  {p['name']:<24} n={p['n']:<5} m={p['m']:<5} "
          f"configs={len(p['configs'])}")
