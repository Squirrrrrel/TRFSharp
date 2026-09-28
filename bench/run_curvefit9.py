"""SciPy side of the nine-parameter ``curve_fit`` cross-check.

Mirrors bench/csharp/CurveFit9.  ``method='trf'`` is passed explicitly:
SciPy's ``curve_fit`` silently switches to MINPACK's ``lm`` for unconstrained
problems, and TRF.Sharp only implements the trust-region reflective method.
"""

from __future__ import annotations

import base64
import json
import platform
import time

import numpy as np
import scipy
from scipy.optimize import curve_fit, least_squares

REPS = 9


def unb64(d):
    raw = base64.b64decode(d["b64"])
    return np.frombuffer(raw, dtype="<f8").reshape(d["shape"])


def gauss3(t, *p):
    z1 = (t - p[1]) / p[2]
    z2 = (t - p[4]) / p[5]
    z3 = (t - p[7]) / p[8]
    return (p[0] * np.exp(-0.5 * z1 * z1)
            + p[3] * np.exp(-0.5 * z2 * z2)
            + p[6] * np.exp(-0.5 * z3 * z3))


def decay_peak(t, *p):
    z = (t - p[5]) / p[6]
    return (p[0] * np.exp(-t / p[1])
            + p[2] * np.exp(-t / p[3])
            + p[4] * np.exp(-0.5 * z * z)
            + p[7] + p[8] * t)


def make_model(name):
    if name.startswith("gauss3"):
        return lambda t, a1, m1, s1, a2, m2, s2, a3, m3, s3: gauss3(
            t, a1, m1, s1, a2, m2, s2, a3, m3, s3)
    return lambda t, a1, t1, a2, t2, ag, mu, sg, c0, c1: decay_peak(
        t, a1, t1, a2, t2, ag, mu, sg, c0, c1)


doc = json.load(open("problems.json", encoding="utf-8"))
wanted = ("gauss3_n9", "decay_peak_n9", "gauss3_outliers_n9")
cases = []

for prob in doc["problems"]:
    name = prob["name"]
    if name not in wanted:
        continue
    t = unb64(prob["data"]["t"])
    y = unb64(prob["data"]["y"])
    x0 = np.array(prob["x0"], float)
    lb = ub = None
    for conf in prob["configs"]:
        if conf["lb"] is not None and conf["ub"] is not None:
            lb = np.array(conf["lb"], float)
            ub = np.array(conf["ub"], float)
            break

    model = make_model(name)
    for bounded in (False, True):
        kw = {"method": "trf"}
        if bounded:
            kw["bounds"] = (lb, ub)
        best = float("inf")
        popt = pcov = None
        for _ in range(REPS):
            t0 = time.perf_counter()
            popt, pcov = curve_fit(model, t, y, p0=x0.copy(), **kw)
            best = min(best, time.perf_counter() - t0)

        # curve_fit discards the solver counters; re-run through least_squares
        # with the identical residual to recover them.
        res = least_squares(
            lambda p: model(t, *p) - y, x0.copy(), method="trf",
            bounds=(lb, ub) if bounded else (-np.inf, np.inf))

        cases.append({
            "case": f"{name}/{'bounded' if bounded else 'unbounded'}",
            "problem": name,
            "bounded": bounded,
            "n": int(x0.size),
            "m": int(t.size),
            "popt": [float(v) for v in popt],
            "pcov_diag": [float(v) for v in np.diag(pcov)],
            "perr": [float(v) for v in np.sqrt(np.diag(pcov))],
            "perr_norm": float(np.linalg.norm(np.sqrt(np.diag(pcov)))),
            "cost": float(res.cost),
            "nfev": int(res.nfev),
            "status": int(res.status),
            "t_min": best,
        })
        print(f"  {name}/{'bounded' if bounded else 'unbounded':<10} "
              f"cost={res.cost:.6e} nfev={res.nfev:3d} status={res.status} "
              f"t={best * 1e3:8.3f} ms")

json.dump({
    "cases": cases,
    "meta": {
        "reps": REPS,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "note": "scipy.optimize.curve_fit(method='trf'), nine-parameter models",
    },
}, open("results_curvefit9_python.json", "w", encoding="utf-8"), indent=1)
print(f"wrote results_curvefit9_python.json: {len(cases)} cases")
