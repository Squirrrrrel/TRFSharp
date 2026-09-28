"""Generate reference results from SciPy's TRF implementation.

Run with the interpreter that has SciPy installed, e.g.
    "D:/Program Files/Python311-arm64/python.exe" gen_reference.py
It writes reference.json next to this file.
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares, curve_fit

EPS = np.finfo(float).eps


def rosen(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


def rosen_jac(x):
    return np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])


# --- deterministic data sets -------------------------------------------------
T = np.linspace(0.0, 10.0, 15)
Y_EXP = 0.5 + 2.0 * np.exp(-1.0 * T)

X_LIN = np.arange(10, dtype=float)
Y_LIN = 1.0 - 2.0 * X_LIN


def exp_res(p):
    return p[0] + p[1] * np.exp(p[2] * T) - Y_EXP


def lin_res(p):
    return p[0] * X_LIN + p[1] - Y_LIN


def rank_deficient(p):
    return np.array([p[0] + p[1] - 1.0, 2 * p[0] + 2 * p[1] - 2.0])


def underdetermined(p):
    return np.array([p[0] * p[1] - 1.0])


def broyden(p):
    f = (3 - p) * p + 1
    f[1:] -= p[:-1]
    f[:-1] -= 2 * p[1:]
    return f


def summarize(res):
    return {
        "x": [float(v) for v in res.x],
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "nfev": int(res.nfev),
        "njev": int(res.njev) if res.njev is not None else -1,
        "status": int(res.status),
        "active_mask": [int(v) for v in res.active_mask],
    }


cases = []


def add(name, fun, x0, **kwargs):
    jac = kwargs.pop("jac", "2-point")
    lb = kwargs.pop("lb", -np.inf)
    ub = kwargs.pop("ub", np.inf)
    res = least_squares(fun, np.array(x0, dtype=float), jac=jac,
                        bounds=(lb, ub), method="trf", **kwargs)
    cases.append({"name": name, "expected": summarize(res)})


# 1-2: Rosenbrock, analytic Jacobian
add("rosen_analytic_unbounded", rosen, [2.0, 2.0], jac=rosen_jac)
add("rosen_analytic_bounded", rosen, [2.0, 2.0], jac=rosen_jac,
    lb=[-np.inf, 1.5], ub=np.inf)

# 3-5: finite-difference Jacobians
add("rosen_fd2_unbounded", rosen, [2.0, 2.0])
add("rosen_fd2_bounded", rosen, [2.0, 2.0], lb=[-np.inf, 1.5], ub=np.inf)
add("rosen_fd3_bounded", rosen, [2.0, 2.0], lb=[-np.inf, 1.5], ub=np.inf,
    jac="3-point")

# 6: exponential fit, interior optimum, bounds inactive
add("exp_fit_interior", exp_res, [1.0, 1.0, 0.0],
    lb=[0.0, 0.0, -5.0], ub=[10.0, 10.0, 5.0])

# 7: same but with bounds forcing the solution onto a face
add("exp_fit_upper_active", exp_res, [0.2, 1.0, 0.0],
    lb=[0.0, 0.0, -5.0], ub=[0.4, 10.0, 5.0])

# 8: linear fit whose unconstrained optimum violates a >= 0
add("linear_fit_at_lower_bound", lin_res, [1.0, 1.0],
    lb=[0.0, -np.inf], ub=np.inf)

# 9-10: scaling
add("rosen_xscale_jac", rosen, [2.0, 2.0], jac=rosen_jac,
    lb=[-np.inf, 1.5], ub=np.inf, x_scale="jac")
add("rosen_xscale_array", rosen, [2.0, 2.0], jac=rosen_jac,
    lb=[-np.inf, 1.5], ub=np.inf, x_scale=[2.0, 0.5])

# 11: robust loss
add("exp_fit_soft_l1", exp_res, [1.0, 1.0, 0.0],
    lb=[0.0, 0.0, -5.0], ub=[10.0, 10.0, 5.0], loss="soft_l1", f_scale=0.1)

# 11b-d: other robust losses
for _loss in ("huber", "cauchy", "arctan"):
    add(f"exp_fit_{_loss}", exp_res, [1.0, 1.0, 0.0],
        lb=[0.0, 0.0, -5.0], ub=[10.0, 10.0, 5.0], loss=_loss, f_scale=0.1)

# 12: rank deficient Jacobian
add("rank_deficient", rank_deficient, [0.5, 0.5],
    lb=[-1.0, -1.0], ub=[1.0, 1.0])

# 13: fewer residuals than variables
add("underdetermined", underdetermined, [1.0, 2.0],
    lb=[0.1, 0.1], ub=[5.0, 5.0])

# 14-15: lsmr trust-region solver
add("rosen_lsmr_bounded", rosen, [2.0, 2.0], jac=rosen_jac,
    lb=[-np.inf, 1.5], ub=np.inf, tr_solver="lsmr")
add("rosen_lsmr_unbounded", rosen, [2.0, 2.0], jac=rosen_jac,
    tr_solver="lsmr")

# 16: start exactly on the bounds (exercises make_strictly_feasible + one-sided FD)
add("exp_fit_x0_on_bounds", exp_res, [0.1, 4.0, -3.0],
    lb=[0.1, 0.5, -3.0], ub=[2.0, 4.0, -0.1])

# 17: larger problem, Broyden tridiagonal, n = 30
add("broyden_fd2_n30", broyden, [-1.0] * 30)

# 18: user supplied diff_step
add("rosen_fd2_diff_step", rosen, [2.0, 2.0], lb=[-np.inf, 1.5], ub=np.inf,
    diff_step=1e-6)

# --- curve_fit ---------------------------------------------------------------
curve_cases = []


def exp_model(t, a, b, c):
    return a + b * np.exp(c * t)


def lin_model(t, a, b):
    return a * t + b


def add_curve(name, model, xdata, ydata, p0, lb, ub):
    popt, pcov = curve_fit(model, xdata, ydata, p0=p0, bounds=(lb, ub))
    curve_cases.append({
        "name": name,
        "popt": [float(v) for v in popt],
        "pcov_diag": [float(pcov[i, i]) for i in range(len(popt))],
        "pcov": [[float(v) for v in row] for row in pcov],
    })


add_curve("curve_exp", exp_model, T, Y_EXP, [1.0, 1.0, 0.0],
          [0.0, 0.0, -5.0], [10.0, 10.0, 5.0])
add_curve("curve_lin_clamped", lin_model, X_LIN, Y_LIN, [1.0, 1.0],
          [0.0, -np.inf], [np.inf, np.inf])

out = {
    "scipy_version": __import__("scipy").__version__,
    "numpy_version": np.__version__,
    "least_squares": cases,
    "curve_fit": curve_cases,
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)

print(f"wrote {path}")
for c in cases:
    e = c["expected"]
    print(f"  {c['name']:26s} x={['%.10g' % v for v in e['x']]} cost={e['cost']:.10g} "
          f"nfev={e['nfev']} njev={e['njev']} status={e['status']}")
for c in curve_cases:
    print(f"  {c['name']:26s} popt={['%.10g' % v for v in c['popt']]}")
